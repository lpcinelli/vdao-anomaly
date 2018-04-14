import argparse
import json
import os

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import callbacks
from keras.optimizers import SGD, Adam, Adamax
from keras.utils import multi_gpu_model
import datasets.hdf5_vdao as vdao
from metrics import Distance, FalseNegRate, FalsePosRate, FBetaScore
from archs import mlp

# BLOCOS DE TESTE -- Como nao e possivel treinar com os mesmo objetos do teste, foram feitos
# blocos de teste, onde a rede e treinada com todos os objetos exceto os que estao no bloco
# (Ex: Bloco de teste [0,2,5,6] - treino com [1,3,4,7,8,9,10,11])
VDAO_TEST_ENTRIES_OBJ_NB = [[0, 2, 5, 6], [1, 3, 4, 8], [7, 9, 12, 13], [
    10, 11, 14, 19
], [15, 16, 20, 22], [17, 21, 24, 29], [18, 28, 33, 38], [23, 25, 40, 41], [
    26, 27, 34, 39
], [30, 35, 36, 46], [31, 37, 43, 49], [32, 42, 44, 47], [45, 50, 52, 56],
    [48, 51, 54, 55], [53, 56, 8, 10],
    [57, 11, 18, 24], [58, 13, 23, 46]]

# Nomes das camadas onde foram extraidos os mapas de features
LAYER_NAME = [
    'res2a', 'res2b', 'res2c', 'res3a', 'res3b', 'res3c', 'res3d', 'res4a',
    'res4b', 'res4c', 'res4d', 'res4e', 'res4f', 'res5a', 'res5b', 'res5c',
    'avg_pool'
]
LAYER_NAME = [
    name + '_branch2a' for name in LAYER_NAME if name.startswith('res')
]

optimizers = {'adam': Adam, 'adamax': Adamax, 'sgd': SGD}


def train(args):
    # print(K.tensorflow_backend._get_available_gpus())
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = ''
    K.set_session(tf.Session(config=config))
    # print('Available GPU: {}'.format(K.tensorflow_backend._get_available_gpus()))
    print('save models and logs to dir: {}'.format(args.save_dir))

    # optimizer
    optimizer = optimizers[args.optim.lower()](lr=args.lr)

    # learning rate schedule
    def schedule(epoch, lr): return lr * \
        args.lr_factor**(epoch // args.lr_span)
    lr_scheduler = callbacks.LearningRateScheduler(schedule)

    # Setting dataset helper
    database = vdao.VDAO(args.dataset_dir, args.train_file, args.test_file,
        val_ratio=args.val_ratio, aloi_file=args.aloi_file)

    cross_history = {}
    for layer in LAYER_NAME:
        seen_vids = []
        database.set_layer(layer)
        aloi_samples = database.loadData(mode='aloi')
        cross_history[layer] = {'train': {}, 'val': {}, 'test': {}}

        for test_idx, test_subset in enumerate(VDAO_TEST_ENTRIES_OBJ_NB):

            layer_path = os.path.join(args.save_dir, layer)

            outside_training = [vid for vid in test_subset if vid in seen_vids]
            seen_vids += test_subset

            # Load data
            train_samples, test_samples = database.loadData(
                test_subset, exclude_vids=outside_training)
            train_samples, val_samples = database.splitValidation()

            train_samples = [np.concatenate((simple, aloi)) for simple, aloi \
                in zip(train_samples, aloi_samples)]

            # Create model
            model = mlp.mlp(
                input_shape=train_samples[0].shape[1:],
                neurons_per_layer=args.nb_neurons,
                weight_decay=args.weight_decay)

            if args.multi_gpu:
                try:
                    model = multi_gpu_model(model, gpus=2)
                except Exception as e:
                    print('Not possible to run on multiple GPUs\nError: {}'.format(e))

            # Instantiate metrics
            dis = Distance()
            fnr = FalseNegRate()
            fpr = FalsePosRate()
            f1 = FBetaScore(beta=1)

            # Glue everything together
            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy', fnr, fpr, dis, f1])

            if not os.path.exists(layer_path):
                os.makedirs(layer_path)

            checkpointer = callbacks.ModelCheckpoint(
                os.path.join(layer_path,
                             'model.test{:02d}-ep{{epoch:02d}}.pth'.format(
                                 test_idx)))
            csv_logger = callbacks.CSVLogger(
                os.path.join(layer_path,
                             'training.test{:02d}.log'.format(test_idx)))

            # Train the model
            history = model.fit(
                x=train_samples[0],
                y=train_samples[1],
                batch_size=args.batch_size,
                epochs=args.epochs,
                shuffle=True,
                verbose=1,
                callbacks=[lr_scheduler, csv_logger, checkpointer],
                validation_data=val_samples)

            print('\nFinished training {}/{}'.format(
                test_idx+1, len(VDAO_TEST_ENTRIES_OBJ_NB)))

            for name, meter in history.history.items():
                # should I log the highes or the latest value here? (stdout)
                try:
                    mode = 'val' if name.startswith('val') else 'train'
                    name = name.split('_')[-1]
                    cross_history[layer][mode][name] += [float(meter[-1])]
                except KeyError:
                    cross_history[layer][mode][name] = [float(meter[-1])]

            # Evaluate on TEST set
            results = model.evaluate(
                x=test_samples[0],
                y=test_samples[1],
                batch_size=args.batch_size,
                verbose=0,
                sample_weight=None)

            for idx, meter in enumerate(results):
                try:
                    cross_history[layer]['test'][model.metrics_names[idx]] += [float(meter)]
                except:
                    cross_history[layer]['test'][model.metrics_names[idx]] = [float(meter)]

            msg = ['TEST:: ']
            msg += [
                '{0}: {1}  '.format(
                    name, meter[-1]) for name, meter in
                cross_history[layer]['test'].items()
            ]
            print(''.join(msg))


        for mode, history in cross_history[layer].items():
            for name, meter in history.items():
                cross_history[layer][mode][name] += ['avg: {}'.format(
                    np.asarray(meter).mean())]

        msg = ['\nLAYER: {}  -->  '.format(layer)]
        msg += [
            '{0}: {1}  '.format(
                name, meter[-1].split()[-1][:6]) for name, meter in
            cross_history[layer]['test'].items()
        ]
        print(''.join(msg))
        print('\n' + '* ' * 80 + '\n\n')

    del args.func
    cross_history['config'] = vars(args)
    with open(os.path.join(args.save_dir, 'summary.json'), 'w') as fp:
        json.dump(cross_history, fp, sort_keys=True, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly detection in VDAO')
    subparsers = parser.add_subparsers()
    # Loaders
    parser.add_argument(
        '--dataset-dir',
        '--dir',
        default='/home/bruno.afonso/datasets/article_HDF5',
        type=str,
        metavar='PATH',
        help='The path to the dataset dir')
    parser.add_argument(
        '-b',
        '--batch-size',
        default=128,
        type=int,
        metavar='N',
        help='mini-batch size (default: 32)')
    # Optimizer
    parser.add_argument(
        '--optim',
        default='adamax',
        type=str,
        help='optimizers: ' + ' | '.join(sorted(optimizers.keys())) +
        ' (default: adamax)')
    parser.add_argument(
        '--multi-gpu',
        '--multiple-gpu',
        action='store_true',
        help='Run on several GPUs if available')

    # Train parser
    tr_parser = subparsers.add_parser('train', help='Network training')

    tr_parser.add_argument(
        '--save',
        dest='save_dir',
        type=str,
        default='models/',
        help='name of the saved model')
    tr_parser.add_argument(
        '--epochs',
        default=20,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    tr_parser.add_argument(
        '--val-ratio',
        default=0.1,
        type=float,
        metavar='N',
        help='Val set relative to train size (ratio)')
    # tr_parser.add_argument(
    #     '--aloi',
    #     action='store_true',
    #     help='Adds ALOI-augmented data to training')
    tr_parser.add_argument(
        '--test-file',
        type=str,
        metavar='PATH',
        default='59_videos_test_batch.h5',
        help='Path to test file'
    )
    tr_parser.add_argument(
        '--train-file',
        type=str,
        metavar='PATH',
        default='train_batch_VDAO.h5',
        help='Path to train file',
    )
    tr_parser.add_argument(
        '--aloi-file',
        type=str,
        metavar='PATH',
        default='train_batch_ALOI.h5',
        help='Path to ALOI-augmented imgs file'
    )
    # Architecture parameters
    tr_parser.add_argument(
        '--nb-neurons',
        '--neurons',
        '--hidden-layers',
        '--hidden',
        nargs='+',
        type=int,
        default=[50, 1600],
        metavar='NB_NEURONS',
        help='List of hidden layers sizes')

    # Hyperparameters
    tr_parser.add_argument(
        '--lr',
        '--learning-rate',
        default=2e-3,
        type=float,
        metavar='LR',
        help='initial learning rate')
    tr_parser.add_argument(
        '--lr-factor',
        '--learning-decay',
        default=1,
        type=float,
        metavar='LRF',
        help='learning rate decay factor')
    tr_parser.add_argument(
        '--lr-span',
        '--lr-time',
        default=10,
        type=float,
        metavar='LRS',
        help='time span for each learning rate step')
    tr_parser.add_argument(
        '--weight-decay',
        '--wd',
        default=0,
        type=float,
        metavar='W',
        help='weight decay (default: 0)')
    tr_parser.set_defaults(func=train)

    args = parser.parse_args()
    print(args)
    args.func(args)

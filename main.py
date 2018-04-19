import argparse
import glob
import json
import os
import pdb
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import datasets.hdf5_vdao as vdao
import keras
import metrics
import tensorflow as tf
import utils
from archs import mlp
from keras import backend as K
from keras import callbacks
from keras.optimizers import SGD, Adam, Adamax
from keras.utils import multi_gpu_model

mpl.use('Agg')

LAYER_NAME = [
    'res2a', 'res2b', 'res2c', 'res3a', 'res3b', 'res3c', 'res3d', 'res4a',
    'res4b', 'res4c', 'res4d', 'res4e', 'res4f', 'res5a', 'res5b', 'res5c',
    'avg_pool'
]
# LAYER_NAME = [
#     'res2a',
# ]
LAYER_NAME = [
    name + '_branch2a' for name in LAYER_NAME if name.startswith('res')
]

optimizers = {'adam': Adam, 'adamax': Adamax, 'sgd': SGD}


def predict(args):
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = ''
    K.set_session(tf.Session(config=config))
    print('save results: {}'.format(args.save_dir))

    # Setting dataset helper
    database = vdao.VDAO(args.dataset_dir, args.train_file, args.test_file)

    # Useful metrics to record
    metrics_list = [metrics.fnr, metrics.fpr, metrics.distance, metrics.f1,
                    metrics.tp, metrics.tn, metrics.fp, metrics.fn]
    metrics_names = [metric.__name__ for metric in metrics_list]

    cross_avgs = {}
    cross_best = {}
    for layer in LAYER_NAME:
        database.set_layer(layer)
        cross_avgs[layer] = {}
        test_sizes = []
        meters_test = []
        thres_vals = np.arange(0.01, 1.0, 0.01)

        for test_idx, (_, _, test_samples, test_size) \
                in enumerate(database.load_generator(val_set=False)):
            test_sizes.append(test_size)

            model_name = os.path.join(
                args.model_path, layer, 'model.test{:02d}-ep'.format(test_idx))
            model_name = glob.glob(model_name + '*')[-1]
            model = keras.models.load_model(model_name, compile=False)

            if args.multi_gpu:
                try:
                    model = multi_gpu_model(model)
                except Exception as e:
                    print('Not possible to run on multiple GPUs\n'
                          'Error: {}'.format(e))

            # DOUBT: Does model.evaluate update stateful metric's states?
            probas_ = model.predict_proba(
                test_samples[0],
                batch_size=args.batch_size,
                verbose=0).squeeze()

            meters_test += [metrics.compose(metrics_list,
                                            (test_samples[1], probas_),
                                            thres_vals.tolist())]

        df = pd.DataFrame(meters_test, columns=metrics_names)
        for col in df.columns:
            cross_avgs[layer][col] = np.average(np.stack(df[col]),
                                                weights=test_sizes, axis=0)

        best_idx = np.argmin(cross_avgs[layer]['distance'])

        cross_best[layer] = {name: meter[best_idx] for name,
                             meter in cross_avgs[layer].items()}
        cross_best[layer]['threshold'] = thres_vals[best_idx]
        print(cross_best)

    utils.save_data({'all': cross_avgs, 'best': cross_best}, args.save_dir)


def train(args):
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = ''
    K.set_session(tf.Session(config=config))
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

    # Useful metrics to record
    metrics_list = [metrics.fnr, metrics.fpr, metrics.distance, metrics.f1,
                    metrics.tp, metrics.tn, metrics.fp, metrics.fn]
    metrics_names = [metric.__name__ for metric in metrics_list]

    cross_history = {}
    for layer in LAYER_NAME:
        # seen_vids = []
        test_sizes = []
        database.set_layer(layer)
        # aloi_samples = database.load_data(mode='aloi')
        cross_history[layer] = {'train': {}, 'val': {}, 'test': {}}
        roc = metrics.ROC()
        layer_path = os.path.join(args.save_dir, layer)

        for test_idx, (train_samples, val_samples, test_samples, test_size) \
                in enumerate(database.load_generator()):
            # for test_idx, test_subset in enumerate(VDAO_TEST_ENTRIES_OBJ_NB):

            # outside_training = [vid for vid in test_subset if vid in seen_vids]
            # seen_vids += test_subset

            # # Load data
            # train_samples, test_samples = database.load_data(
            #     test_subset, exclude_vids=outside_training)
            # train_samples, val_samples = database.split_validation(mode='video')

            # if args.aloi_file is not None:
            #     train_samples = tuple(np.concatenate((simple, aloi)) for simple, \
            #         aloi in zip(train_samples, aloi_samples))

            test_sizes.append(test_size)

            # Create model
            model = mlp.mlp(
                input_shape=train_samples[0].shape[1:],
                neurons_per_layer=args.nb_neurons,
                weight_decay=args.weight_decay)

            if args.multi_gpu:
                try:
                    model = multi_gpu_model(model)
                except Exception as e:
                    print('Not possible to run on multiple GPUs\n'
                          'Error: {}'.format(e))

            # Glue everything together
            model.compile(
                loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy',
                         metrics.FalseNegRate(),
                         metrics.FalsePosRate(),
                         metrics.Distance(),
                         metrics.FBetaScore(beta=1),
                         metrics.TruePos(),
                         metrics.TrueNeg(),
                         metrics.FalsePos(),
                         metrics.FalseNeg()])

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

            print('\nFinished training {}'.format(test_idx+1))

            for name, meter in history.history.items():
                # should I log the highest or the latest value here? (stdout)
                try:
                    mode = 'val' if name.startswith('val') else 'train'
                    name = name.split('_')[-1]
                    cross_history[layer][mode][name] += [float(meter[-1])]
                except KeyError:
                    cross_history[layer][mode][name] = [float(meter[-1])]

            probas_ = model.predict_proba(
                val_samples[0],
                batch_size=args.batch_size,
                verbose=0).squeeze()

            # Compute ROC curve and find thresold that minimizes dist
            best_threshold, best_dist = roc(val_samples[1], probas_)

            # Evaluate on VAL set (threshold @ 50%. @ `best_threshold`)
            thres_vals = [0.5, best_threshold]
            meters_val = metrics.compose(metrics_list,
                                         (val_samples[1], probas_), thres_vals)

            for name, meters in zip(metrics_names, meters_val):
                try:
                    cross_history[layer]['val']['pp_' + name] += [[
                        float(meter) for meter in meters]]
                except:
                    cross_history[layer]['val']['pp_' + name] = [[
                        float(meter) for meter in meters]]

            # Print validation results w/ thresholds other than 0.5
            msg = ['VAL:: thresholds @ {}\n'.format(thres_vals[1:])]
            msg += [
                '{0}: {1}  \n'.format(
                    name, meter[-1][1:] if isinstance(meter[-1], (list, tuple))
                    else meter[-1]) for name, meter in
                cross_history[layer]['val'].items() if name is not 'thresholds'
            ]
            print(''.join(msg))

            # DOUBT: Does model.evaluate update stateful metric's states?
            probas_ = model.predict_proba(
                test_samples[0],
                batch_size=args.batch_size,
                verbose=0).squeeze()

            # Evaluate on TEST set (threshold @ 50%. @ `best_threshold`)
            meters_test = metrics.compose(metrics_list,
                                          (test_samples[1], probas_), thres_vals)

            for name, meters in zip(metrics_names, meters_test):
                try:
                    cross_history[layer]['test'][name] += [[
                        float(meter) for meter in meters]]
                except:
                    cross_history[layer]['test'][name] = [[
                        float(meter) for meter in meters]]

            # Print test results w/ all thresholds
            msg = ['\nTEST:: thresholds @ {}\n'.format(thres_vals)]
            msg += [
                '{0}: {1}  \n'.format(
                    name, meter[-1]) for name, meter in
                cross_history[layer]['test'].items() if name is not 'thresholds'
            ]
            print(''.join(msg))

            thres_vals = [[float(thres) for thres in thres_vals]]
            for mode in ['val', 'test']:
                try:
                    cross_history[layer][mode]['thresholds'] += thres_vals
                except:
                    cross_history[layer][mode]['thresholds'] = thres_vals

        for mode, history in cross_history[layer].items():
            for name, meter in history.items():
                if name is 'thresholds':
                    continue
                cross_history[layer][mode][name] += [
                    np.average(meter, weights=test_sizes, axis=0).astype(
                        'float64').tolist()]

        cross_history[layer]['test']['nb_vids'] = test_sizes

        mean_tpr, mean_auc = [meter.astype(
            'float64').tolist() for meter in roc.mean()]
        std_tpr, std_auc = [meter.astype('float64').tolist()
                            for meter in roc.std()]
        cross_history[layer]['val']['roc'] = {
            'interp_tprs-mean': mean_tpr,
            'auc-mean': mean_auc,
            'interp_tprs-std': std_tpr,
            'auc-std': std_auc}
        roc.plot(os.path.join(layer_path, 'roc-crossval.eps'))

        # Print layer avg results @50% and @`best_threshold`
        msg = ['\nLAYER: {} :: thresholds @ {}\n'.format(layer,
                                                         cross_history[layer]['test']['thresholds'])]
        msg += [
            '| {0}: '.format(name) + ('{:.4f}  '*int(len(meter[-1]))).format(
                *meter[-1]) for name, meter in
            cross_history[layer]['test'].items() if name not in ['thresholds', 'nb_vids']
        ]
        print(''.join(msg))
        print('\n' + '* ' * 80 + '\n\n')

    # Plot mean ROC curve for each layer
    for layer, info in cross_history.items():
        auc_mean = info['val']['roc']['auc-mean']
        auc_std = info['val']['roc']['auc-std']
        plt.plot(roc.mean_fpr, info['val']['roc']['interp_tprs-mean'],
                 lw=2, alpha=0.8, label=r'ROC - {} (AUC = {:.2f} $\pm$ {:0.2f})'.format(
            layer.split('_')[0], auc_mean, auc_std))
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1,
             color='r', label='Identidade', alpha=.8)
    roc.label_plot()
    plt.savefig(os.path.join(args.save_dir, 'mean-roc.eps'))
    plt.close()

    # saving quick access summary w/ results
    del args.func
    cross_history['config'] = vars(args)

    utils.save_data(cross_history, os.path.join(args.save_dir, 'summary'))


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
    parser.add_argument(
        '--save',
        dest='save_dir',
        type=str,
        default='models/',
        help='path to save results to')

    # Train parser
    tr_parser = subparsers.add_parser('train', help='Network training')

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
        default=None,
        help='Path to ALOI-augmented imgs file'
        # 'train_batch_ALOI.h5'
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

    # Predict parser
    pred_parser = subparsers.add_parser('predict', help='Network training')

    pred_parser.add_argument(
        '--test-file',
        type=str,
        metavar='PATH',
        default='59_videos_test_batch.h5',
        help='Path to test file')

    pred_parser.add_argument(
        '--train-file',
        type=str,
        metavar='PATH',
        default='train_batch_VDAO.h5',
        help='Path to train file')
    pred_parser.add_argument(
        '--load',
        dest='model_path',
        type=str,
        metavar='PATH',
        help='Path to trained model dir')

    pred_parser.set_defaults(func=predict)

    args = parser.parse_args()
    print(args)
    args.func(args)

import argparse
import glob
import json
import os
import pdb
import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import archs
import datasets.hdf5_vdao as vdao
import keras
import metrics
import tensorflow as tf
import utils
from datasets.hdf5_vdao import LAYER_NAME
from keras import backend as K
from keras import callbacks
from keras.optimizers import SGD, Adam, Adamax
from keras.utils import multi_gpu_model

import pdb

optimizers = {'adam': Adam, 'adamax': Adamax, 'sgd': SGD}

arch_names = sorted(name for name in archs.__dict__
                    if name.islower() and not name.startswith("__")
                    and callable(archs.__dict__[name]))


def _common(args, func, **kwargs):
    mode = 'train' if func.__name__ == '_train' else 'test'
    if mode == 'test':
        # Instatiate dataloader
        validation = False
        database = vdao.VDAO(args.dataset_dir, args.file, mode=mode,
                            val_set=validation)
    else:
        validation =  args.val_roc
        database = vdao.VDAO(args.dataset_dir, args.file, mode=mode,
                            val_set=validation, val_ratio=args.val_ratio,
                            aloi_file=args.aloi_file)

    # Set tensorflow session configurations
    config = tf.ConfigProto()
    config.gpu_options.visible_device_list = ''
    K.set_session(tf.Session(config=config))
    print('save results: {}'.format(args.save_dir))


    # Useful metrics to record
    metrics_list = [metrics.fnr, metrics.fpr, metrics.distance, metrics.f1,
                    metrics.tp, metrics.tn, metrics.fp, metrics.fn]
    meters = {func.__name__: func for func in metrics_list}

    thresholds = kwargs.pop('thresholds', None)

    logger = {}
    # Apply func to data comming from all specified layers
    for layer in LAYER_NAME:
        print('layer: {}'.format(layer))
        database.set_layer(layer)
        layer_path = os.path.join(args.save_dir, layer)
        cross_history = utils.History()
        outputs = []

        roc = metrics.ROC() if validation is True else None

        # Apply func to each partition of the data
        for group_idx, (samples, set_size) in enumerate(
                database.load_generator(
                    **utils.parse_kwparams(args.cv_params),
                    inner_kwargs=utils.parse_kwparams(args.inner_val_params))):

            # Load old model or create a new one
            if args.load_model is not None:
                model_name = os.path.join(args.load_model, layer,
                                          'model.test{:02d}-ep'.format(group_idx))
                model_name = glob.glob(model_name + '*')[-1]
                model = keras.models.load_model(model_name, compile=False)
            else:
                model = archs.__dict__[args.arch](
                    input_shape=samples[0][0].shape[1:],
                    weight_decay=args.weight_decay,
                    **utils.parse_kwparams(args.arch_params))

            # Make it run on mutiple GPUs (batchwise)
            if args.multi_gpu:
                try:
                    model = multi_gpu_model(model)
                except Exception as e:
                    print('Not possible to run on multiple GPUs\n'
                          'Error: {}'.format(e))

            if mode == 'train':
                output = func(args, model, samples, set_size, meters, layer_path,
                            group_idx, cross_history,  roc=roc)
            else:
                if type(thresholds) is dict:
                    group_thresholds = thresholds[layer][group_idx]
                else:
                    group_thresholds = thresholds

                output = func(args, model, samples, set_size, meters,
                             threshold=group_thresholds)

            outputs += [output]

        if mode == 'train':
            logger[layer] = {'history': cross_history}
            if roc is not None:
                logger[layer].update({'roc': roc})
        else:
            logger[layer] = {'output': outputs}

        print('\n' + '* ' * 80 + '\n\n')

    return logger


def _eval(args, model, test_samples, set_size, meters, threshold=0.5):
    measures = []
    set_size = np.hstack((np.zeros(1, dtype='int64'), np.cumsum(set_size['test'])))
    data, labels = test_samples
    for start, stop in zip(set_size[:-1], set_size[1:]):
        measurements, _ = _evaluate(model, (data[start:stop], labels[start:stop]), meters,
                                    tune_threshold=False, batch_size=args.batch_size,
                                    thresholds=threshold, mode='test', verbose=0)
        measures += [measurements]
    return measures


def eval(args):
    # use common and _evaluate
    # Check if _evaluate is still correct after changings
    # Inspect results: they seem far too good to be true
    checkpoint = pickle.load(open(os.path.join(
                                    args.load_model,'summary.pkl'), 'rb'))
    thresholds = {}
    for key, val in checkpoint.items():
        if key == 'config':
            continue
        try:
            thres = val['history'].history['val']['thresholds']
            # thres = val['val']['thresholds']
            thresholds.update({key: np.asarray(thres)[:, 1]})
        except KeyError:
            print('No optimized threshold found')
            thresholds = None

    logger = _common(args, _eval, thresholds=thresholds)
    logger = {layer: results['output'] for layer, results in logger.items()}
    utils.save_data(logger, os.path.join(args.load_model, 'test-results'),
            json_format=False, pickle_format=True)


def _train(args, model, samples, set_size, meters, layer_path, group_idx,
        cross_history, roc=None, **kwargs):

    # optimizer
    optimizer = optimizers[args.optim.lower()](lr=args.lr)

    # learning rate schedule
    def schedule(epoch, lr): return lr * \
        args.lr_factor**(epoch // args.lr_span)
    lr_scheduler = callbacks.LearningRateScheduler(schedule)

    train_samples, val_samples = samples
    for mode, size in set_size.items():
        if size == 0:
            continue
        cross_history.update('nb_vids', size, mode=mode)

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

    # Should I monitor here the best val_loss or the metrics of interest?
    # If not all samples are used in an epoch val_loss is noisy
    checkpointer = callbacks.ModelCheckpoint(
        os.path.join(layer_path,
                     'model.test{:02d}-ep{{epoch:02d}}.pth'.format(
                         group_idx)),
        monitor='val_loss', save_best_only=True, mode='min')
    csv_logger = callbacks.CSVLogger(
        os.path.join(layer_path,
                     'training.test{:02d}.log'.format(group_idx)))

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

    print('\nFinished training {}'.format(group_idx+1))

    # TODO: log the best value and not the latest
    for name, meter in history.history.items():
        mode = 'val' if name.startswith('val') else 'train'
        cross_history.update(name, meter[-1], mode)

    if val_samples is not None:
        measurements, thres_vals = _evaluate(
            model, (val_samples, set_size['val']), meters, mode='val',
            batch_size=args.batch_size, tune_threshold=True, roc=roc)

        for name, measures in measurements.items():
            cross_history.update(
                'pp_' + name, [measure for measure in measures], mode='val')

        cross_history.update('thresholds', thres_vals, mode='val')
        return {'history': cross_history, 'roc': roc}

    return {'history': cross_history}


def _evaluate(model, data, meters, batch_size=32, verbose=1, mode='val',
              tune_threshold=False, thresholds=[0.5], roc=None, history=None):
    if mode not in ['val', 'test']:
        raise ValueError('mode should be either val or test')
    if mode is 'test' and tune_threshold is True:
        raise ValueError(
            'You cannot tune the threshold on the test data silly, that is cheating')
    if tune_threshold is True and (roc is None or isinstance(roc, metrics.ROC) is False):
        # Optionally I can create a metrics.ROC object here
        raise ValueError(
            'roc should be an instance of metrics.ROC to optimize threshold')
    if history is not None and isinstance(history, utils.History) is False:
        raise ValueError('history should be an instance of utils.History')

    try:
        (samples, labels), set_size = data
    except ValueError:
        samples, labels = data
        set_size = None

    probas_ = model.predict_proba(
        samples, batch_size=batch_size, verbose=0).squeeze()

    # Compute ROC curve and find thresold that minimizes dist
    if tune_threshold:
        best_threshold, _ = roc(labels, probas_)
        thresholds = thresholds + [best_threshold]

    # Evaluate on VAL set (threshold @ 50%. @ `best_threshold`)
    meters_val = metrics.compose(meters.values(), (labels, probas_),
                                 threshold=thresholds)

    if history:
        for name, measures in zip(meters.keys(), meters_val):
            history.update(
                'pp_' + name, [measure for measure in measures], mode=mode)
        # history.update('thresholds', thresholds, mode=mode)

    if verbose:
        msg = ['\n{}:: thresholds @ {}\n'.format(mode.upper(), thresholds)]
        msg += ['{0}: {1}\n'.format(name, measures) for name, measures in
                zip(meters.keys(), meters_val)]
        print(''.join(msg))

        # utils.print_result(history.history[mode],
        #     '{}:: thresholds @ {}'.format(mode.upper(), thresholds), exclude=['thresholds'])

    return {key: val for key, val in zip(meters.keys(), meters_val)}, thresholds


def train(args):

    logger = _common(args, _train)
    for layer in logger.keys():
        logger[layer]['history'].averages(weights_key='nb_vids',
                                          exclude=['thresholds'])
    # Compute ROC if data available
    if args.val_roc is True:
        roc_stats = {'tprs': {}, 'auc': {}}
        plt.figure('all_layers')
        for layer, results in logger.items():
            roc = results['roc']

            for name, stats in zip(('mean', 'std'), (roc.mean(), roc.std())):
                roc_stats['tprs'][name], roc_stats['auc'][name] = stats

            logger[layer]['roc'] = roc_stats
            roc.plot(os.path.join(os.path.join(args.save_dir, layer),
                                  'roc-crossval.eps'))

            # Plot mean ROC curve for each layer
            plt.figure('all_layers')
            auc_mean, auc_std = roc_stats['auc']['mean'], roc_stats['auc']['std']
            plt.plot(roc.mean_fpr, roc_stats['tprs']['mean'], lw=2, alpha=0.8,
                     label=r'ROC - {} (AUC = {:.2f} $\pm$ {:0.2f})'.format(
                layer.split('_branch')[0], auc_mean, auc_std))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=1,
                 color='r', label='Identidade', alpha=.8)
        metrics.ROC.label_plot()
        plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.savefig(os.path.join(args.save_dir, 'mean-roc.eps'),
                    bbox_inches='tight')
        plt.close()

    # Save history
    del args.func
    logger['config'] = vars(args)
    utils.save_data(logger, os.path.join(args.save_dir, 'summary'),
                    json_format=False)


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
    parser.add_argument(
        '--file',
        type=str,
        metavar='PATH',
        # default='59_videos_test_batch.h5', -- test
        # default='train_batch_VDAO.h5',     -- train
        help='Path to hdf5 data file')
    parser.add_argument(
        '--load',
        dest='load_model',
        type=str,
        metavar='PATH',
        default=None,
        help='Path to trained model dir')

    # Train parser
    tr_parser = subparsers.add_parser('train', help='Network training')

    tr_parser.add_argument(
        '--epochs',
        default=20,
        type=int,
        metavar='N',
        help='number of total epochs to run')
    tr_parser.add_argument(
        '--val-roc',
        action='store_true',
        help='Whether to split train set on train/val')
    tr_parser.add_argument(
        '--val-ratio',
        default=0.1,
        type=float,
        metavar='N',
        help='Val set size relative to train size (ratio)')
    tr_parser.add_argument(
        '--aloi-file',
        type=str,
        metavar='PATH',
        default=None,
        help='Path to ALOI-augmented imgs file'
    )
    parser.add_argument(
        '--cv-params',
        metavar='PARAMS',
        # default=['method=k_fold', 'n_splits=5'],
        default=['method=leave_one_out'],
        # default=['method=manual', 'filename=legacy/test_folds.json'],
        # default=[],
        nargs='+',
        type=str,
        help='cross validation params, methods: ' + ' | '.join(
            list(vdao.group_fetching.keys())) + ' (default method: leave_one_out)'
    )
    parser.add_argument(
        '--inner-val-params',
        metavar='PARAMS',
        default=['mode=video'],
        nargs='+',
        type=str,
        help='inner validation params (default method: LeaveOneGroupOut)'
    )
    # Architecture
    parser.add_argument(
        '--arch',
        '-a',
        metavar='ARCH',
        default='mlp',
        choices=arch_names,
        help='model architecture: ' + ' | '.join(arch_names) +
        ' (default: mlp)')
    parser.add_argument(
        '--arch-params',
        metavar='PARAMS',
        default=['nb_neurons=[50, 1600]'],
        nargs='+',
        type=str,
        help='model architecture params')

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
    pred_parser = subparsers.add_parser('eval', help='Network training')

    pred_parser.set_defaults(func=eval)

    args = parser.parse_args()
    print(args)
    args.func(args)

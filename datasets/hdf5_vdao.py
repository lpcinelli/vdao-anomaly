import os
from itertools import product as iterprod

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# TODO: Make it read this info from (JSON?) file
# Specifies which object appears in each one of the 59 vids (in order)
T59VIDS_OBJS_LST = [['Dark-Blue_Box'] * 2, ['Shoe'], ['Camera_Box'], [
    'Towel'
], ['White_Jar'], ['Pink_Bottle'], ['Shoe'] * 2, ['Dark-Blue_Box'] * 2,
    ['Camera_Box'] * 2, ['White_Jar'] * 3, ['Brown_Box'] * 3,
    ['Pink_Bottle'] * 3, ['Towel'] * 2, ['Black_Coat'] * 3,
    ['Black_Backpack'] * 6, ['Black_Coat'] * 3,
    ['Dark-Blue_Box'] * 2, ['Camera_Box'] * 3,
    ['White_Jar'] * 2, ['Brown_Box'] * 3, ['Pink_Bottle'] * 3,
    ['Towel'] * 3, ['Black_Coat'] * 3, ['Black_Backpack'] * 4]

T59VIDS_OBJS_LST = [item for sublist in T59VIDS_OBJS_LST for item in sublist]

VDAO_DATABASE_LIGHTING_ENTRIES = np.array(['NORMAL-Light', 'EXTRA-Light'])

VDAO_DATABASE_OBJECT_ENTRIES = list(set([
    name for name in T59VIDS_OBJS_LST
])) + ['Mult_Objs1', 'Mult_Objs2', 'Mult_Objs3']

VDAO_DATABASE_OBJECT_POSITION_ENTRIES = np.array(['POS1', 'POS2', 'POS3'])

# path do HDF5
# nomes dos arquivos HDF5
# HDF5_SRC = '/home/bruno.afonso/datasets/article_HDF5'
# HDF5_TEST = '59_videos_test_batch.h5'
# HDF5_TRAIN = 'train_batch_VDAO.h5'


class VDAO(object):
    def __init__(self, dataset_dir, train_file, test_file, val_ratio=0, aloi_file=None):
        self.dataset_dir = dataset_dir
        self.train_file = train_file
        self.test_file = test_file
        self.aloi_file = aloi_file

        self.train_entries = list(iterprod(
            VDAO_DATABASE_LIGHTING_ENTRIES,
            VDAO_DATABASE_OBJECT_ENTRIES,
            VDAO_DATABASE_OBJECT_POSITION_ENTRIES))

        self.val_ratio = val_ratio
        self.out_layer = None

    def set_layer(self, layer_name):
        self.out_layer = layer_name
        # self._reset_data()

    def _reset_data(self):
        self.data = None
        self.labels = None

    def loadData(self, test_vids=[], mode='simple', exclude_vids=[], verbose=1):
        if mode == 'aloi':
            if self.aloi_file is None:
                    raise NameError('HDF5 for aloi-augmented data not set')
            if verbose > 0:
                print('LOADING ALOI VIDEOS\nLAYER: {}\n'.format(self.out_layer))
            self.aloi_data = _loadFile(
                os.path.join(self.dataset_dir, self.aloi_file),
                self.out_layer, [''])

            return self.aloi_data

        elif mode in ['simple', 'vdao-only', 'original']:
            # get which objects appear in the chosen test set
            if len(test_vids) == 0:
                raise ValueError('No test vid specified')

            test_objs = [T59VIDS_OBJS_LST[vid] for vid in test_vids]
            if verbose > 0:
                print('LOADING DATA\nLAYER: {}\nTEST VIDEOS: {}\n'\
                    'TEST OBJS: {}\nEXCLUDE REPEATED: {}\n'.format(
                        self.out_layer, test_vids, test_objs, exclude_vids))

            test_vids = [['video{}'.format(vid+1)]
                         for vid in test_vids if vid not in exclude_vids]

            # load test batches
            self.test_data = _loadFile(
                os.path.join(self.dataset_dir, self.test_file),
                self.out_layer, test_vids)

            # load train batches
            self.train_data = _loadFile(
                os.path.join(self.dataset_dir, self.train_file),
                self.out_layer, self.train_entries, test_objs)

            return self.train_data, self.test_data

    def splitValidation(self, ratio=None, data=None, labels=None):
        if ratio is None:
            ratio = self.val_ratio
        if data is None or labels is None:
            data, labels = self.train_data
        x_train, x_val, y_train, y_val = train_test_split(
            data, labels, stratify=labels, test_size=ratio)

        return (x_train, y_train), (x_val, y_val)


def _loadFile(basepath, out_layer, vid_name_iter, exceptions=[], verbose=1):
    """ Loads the speciied set on 'basepath' ....
    """
    if 'train' in basepath:
        mode = 'train'
    elif 'test' in basepath:
        mode = 'test'
    else:
        raise NameError('Could not determine mode (\'train\'/\'test\') from '
                        'HDF5 filename \'{}\''.format(basepath))

    if verbose > 1:
        print('{} SET...'.format(mode.upper()))
        print('exceptions:{}\n'.format(exceptions))

    h5_file = h5py.File(basepath, 'r')

    data = []
    labels = []
    for items in vid_name_iter:
        if not isinstance(items, list):
            items = list(items)
        filepath = '_'.join(items + [out_layer])
        filepath = filepath + '_{{}}_{}_SET'.format(mode.upper())
        if any(substr in filepath for substr in exceptions):
            continue
        try:
            data.append(h5_file[filepath.format('X')].value)
            labels.append(h5_file[filepath.format('y')].value)
        except KeyError as exception:
            if mode is 'test':
                raise exception

    data = np.concatenate(data)
    labels = np.concatenate(labels).astype(int)

    h5_file.close()

    return data, labels

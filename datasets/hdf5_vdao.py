import os
from itertools import product as iterprod

import h5py
import numpy as np
from sklearn.model_selection import train_test_split, GroupKFold

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
# VDAO_TEST_ENTRIES_OBJ_NB = [[0, 2, 5, 6], [1, 3, 4, 8], [7, 9, 12, 13]]

# Specifies which object appears in each one of the 59 vids (in order)
VIDS_OBJS = [['Dark-Blue_Box'] * 2, ['Shoe'], ['Camera_Box'], [
    'Towel'
], ['White_Jar'], ['Pink_Bottle'], ['Shoe'] * 2, ['Dark-Blue_Box'] * 2,
    ['Camera_Box'] * 2, ['White_Jar'] * 3, ['Brown_Box'] * 3,
    ['Pink_Bottle'] * 3, ['Towel'] * 2, ['Black_Coat'] * 3,
    ['Black_Backpack'] * 6, ['Black_Coat'] * 3,
    ['Dark-Blue_Box'] * 2, ['Camera_Box'] * 3,
    ['White_Jar'] * 2, ['Brown_Box'] * 3, ['Pink_Bottle'] * 3,
    ['Towel'] * 3, ['Black_Coat'] * 3, ['Black_Backpack'] * 4]

VIDS_OBJS = [item for sublist in VIDS_OBJS for item in sublist]

VDAO_DATABASE_LIGHTING_ENTRIES = ['NORMAL-Light', 'EXTRA-Light']

VDAO_DATABASE_OBJECT_ENTRIES = list(set([
    name for name in VIDS_OBJS
])) + ['Mult_Objs1', 'Mult_Objs2', 'Mult_Objs3']

VDAO_DATABASE_OBJECT_POSITION_ENTRIES = ['POS1', 'POS2', 'POS3']

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
        self._reset_data()
        self.train_entries = list(iterprod(
            VDAO_DATABASE_LIGHTING_ENTRIES,
            VDAO_DATABASE_OBJECT_ENTRIES,
            VDAO_DATABASE_OBJECT_POSITION_ENTRIES))

        self.val_ratio = val_ratio

    def set_layer(self, layer_name):
        self._reset_data()

        self.out_layer = layer_name

    def _reset_data(self):
        self.out_layer = None
        self.train_data, self.aloi_data = [None] * 2
        self.val_data, self.test_data = [None] * 2

    def load_generator(self, n_splits=5, val_set=True):
        x = np.arange(len(VIDS_OBJS))
        group_kfold = GroupKFold(n_splits=n_splits)
        self.load_data(mode='aloi')

        for train_vid_idx, test_vid_idx in group_kfold.split(x, x, VIDS_OBJS):
            test_objs = [VIDS_OBJS[vid] for vid in test_vid_idx]
            test_vids = [['video{}'.format(vid+1)] for vid in test_vid_idx]
            train_vids = [
                k for k in self.train_entries if k[1] not in test_objs]

            self.test_data = _loadFile(
                os.path.join(self.dataset_dir, self.test_file),
                self.out_layer, test_vids)

            if val_set is True:
                train_vids, val_vids = self.split_validation(mode='video',
                                                             data=train_vids,
                                                             random_state=0)
                self.val_data = _loadFile(
                    os.path.join(self.dataset_dir, self.train_file),
                    self.out_layer, val_vids, test_objs)

            self.train_data = _loadFile(
                os.path.join(self.dataset_dir, self.train_file),
                self.out_layer, train_vids, test_objs)

            yield _merge_datasets((self.train_data,
                                   self.aloi_data)), \
                self.val_data, self.test_data, len(test_vid_idx)

    def load_data(self, test_vids=[], mode='simple', exclude_vids=[], verbose=1):
        if mode == 'aloi':
            if self.aloi_file is None:
                print('HDF5 for aloi-augmented data not set')
                return
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

            test_objs = [VIDS_OBJS[vid] for vid in test_vids]
            if verbose > 0:
                print('LOADING DATA\nLAYER: {}\nTEST VIDEOS: {}\n'
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

    def split_validation(self, mode='frame', ratio=None, data=None, labels=None, random_state=None):
        extra_data = None
        if ratio is None:
            ratio = self.val_ratio

        if mode is 'frame':
            if data is None:
                data, labels = self.train_data
                extra_data = self.aloi_data

            x_train, x_val, y_train, y_val = train_test_split(
                data, labels, stratify=labels, test_size=ratio, random_state=random_state)
            train_data = x_train, y_train
            val_data = x_val, y_val

            return _merge_datasets((train_data, extra_data)), val_data

        elif mode is 'video':
            if data is None:
                data = np.asarray(self.train_entries)

            x_train, x_val, _, _ = train_test_split(
                data, np.ones(len(data)), test_size=ratio, random_state=random_state)

            if labels is None:
                return x_train, x_val
        else:
            raise ValueError('mode should be either \'frame\' or \'video\'')


def _merge_datasets(datasets):
    """ Merge (concatenates along samples dimension) together datasets
        (data, label).
    Args:
        datasets (list): Contains (data, label) elements to be merged together
    Returns:
        A single dataset tuple (data, label) consisting of all input datasets
        in the same exact order.
    """
    return tuple(np.concatenate(data) for data in zip(*tuple(filter(None, datasets))))


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

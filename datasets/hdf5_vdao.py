import os
from itertools import product
import json

import h5py
import numpy as np
import sklearn.model_selection
from sklearn.model_selection import train_test_split, GroupKFold, LeavePGroupsOut, LeaveOneGroupOut, GroupShuffleSplit

LAYER_NAME = [
    'res2a', 'res2b', 'res2c', 'res3a', 'res3b', 'res3c', 'res3d', 'res4a',
    'res4b', 'res4c', 'res4d', 'res4e', 'res4f', 'res5a', 'res5b', 'res5c',
]
LAYER_NAME = [
    name + '_branch2a' for name in LAYER_NAME if name.startswith('res')
]

# BLOCOS DE TESTE -- Como nao e possivel treinar com os mesmo objetos do teste, foram feitos
# blocos de teste, onde a rede e treinada com todos os objetos exceto os que estao no bloco
# (Ex: Bloco de teste [0,2,5,6] - treino com [1,3,4,7,8,9,10,11])
VDAO_TEST_ENTRIES = [[0, 2, 5, 6], [1, 3, 4, 8], [7, 9, 12, 13], [
    10, 11, 14, 19
], [15, 16, 20, 22], [17, 21, 24, 29], [18, 28, 33, 38], [23, 25, 40, 41], [
    26, 27, 34, 39
], [30, 35, 36, 46], [31, 37, 43, 49], [32, 42, 44, 47], [45, 50, 52, 56],
    [48, 51, 54, 55], [53, 56, 8, 10],
    [57, 11, 18, 24], [58, 13, 23, 46]]

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

VDAO_ILLU = ['NORMAL-Light', 'EXTRA-Light']

VDAO_OBJS = set([
    name for name in VIDS_OBJS + ['Mult_Objs1', 'Mult_Objs2', 'Mult_Objs3']
])

VDAO_POS = ['POS1', 'POS2', 'POS3']


class ManualGroupSplit(object):
    """ Manually defines a cross-validation group split specified by a .json
    file containing. Compatibility class due to Bruno's group partitioning.
    """

    def __init__(self, filename=None, n_splits=None):
        """
        Keyword Arguments:
            filename {string} -- Path to file containing the splits and their videos (default: {None})
            n_splits {int} -- Number of different groups to consider (default: {None})

        Raises:
            ValueError -- Incorret number of splits requested (max: 17)
        """
        if not filename.endswith('.json'):
            filename += '.json'
        with open(filename, 'r') as fp:
            self.test_folds = json.load(fp)

        self.nb_groups = n_splits or len(self.test_folds)
        if self.nb_groups > len(self.test_folds):
            raise ValueError('n_splits should be not greater than {}'.format(
                len(self.test_folds)))

    def split(self, X, y, groups):
        seen_vids = []
        for test_idx in self.test_folds[:self.nb_groups]:

            test_groups = [groups[idx] for idx in test_idx]
            train_idx = [idx for idx, obj in enumerate(
                groups) if obj not in test_groups]

            test_idx = [idx for idx in test_idx if idx not in seen_vids]
            seen_vids += test_idx

            yield train_idx, test_idx


group_fetching = {'k_fold': GroupKFold,
                  'leave_p_out': LeavePGroupsOut,
                  'leave_one_out': LeaveOneGroupOut,
                  'shuffle_split': GroupShuffleSplit,
                  'manual': ManualGroupSplit
                  }


class VDAO(object):
    """[summary]

    Arguments:
        object {[type]} -- [description]

    Raises:
        ValueError -- [description]
        ValueError -- [description]
        ValueError -- [description]

    Returns:
        [type] -- [description]
    """

    def __init__(self, dataset_dir, filename, val_ratio=0,
                 aloi_file=None, val_set=True, mode='train'):
        """[summary]

        Arguments:
            dataset_dir {[type]} -- [description]
            filename {[type]} -- [description]

        Keyword Arguments:
            val_ratio {int} -- [description] (default: {0})
            aloi_file {[type]} -- [description] (default: {None})
            val_set {bool} -- [description] (default: {True})
            mode {str} -- [description] (default: {'train'})
        """

        self.dataset_dir = dataset_dir
        self.files = {mode: filename, 'aloi': aloi_file}
        self.data = {mode: None, 'aloi': None}
        if mode == 'train':
            self.data['val'] = None

        self.train_entries = list(product(VDAO_ILLU, VDAO_OBJS, VDAO_POS))
        self.val_ratio = val_ratio
        self.val_set = val_set
        self.mode = mode

    def set_layer(self, layer_name):
        self.data = {key: None for key in self.data.keys()}
        self.out_layer = layer_name

    def load_generator(self,
                       method='leave_one_out',
                       inner_kwargs=None,
                       **kwargs):
        """[summary]

        Keyword Arguments:
            method {str} -- [description] (default: {'leave_one_out'})
            inner_kwargs {[type]} -- [description] (default: {None})
        """

        x = np.arange(len(VIDS_OBJS))
        fold_method = group_fetching[method](**kwargs)
        mode = inner_kwargs.pop('mode', 'video')

        if mode == 'aloi' and self.files['aloi'] is not None:
            self.data['aloi'], aloi_size = _loadFile(
                    os.path.join(self.dataset_dir, self.files['aloi']),
                    self.out_layer, [''])

        for train_vid_idx, test_vid_idx in fold_method.split(x, x, VIDS_OBJS):
            test_objs = [VIDS_OBJS[vid] for vid in test_vid_idx]
            test_vids = [['video{}'.format(vid+1)] for vid in test_vid_idx]
            train_vids = list(product(VDAO_ILLU, set(
                [VIDS_OBJS[idx] for idx in train_vid_idx]), VDAO_POS))

            if self.mode == 'test':
                self.data['test'], test_size = _loadFile(
                    os.path.join(self.dataset_dir, self.files['test']),
                    self.out_layer, test_vids)
                yield self.data['test'], {'test': test_size}

            else:
                self.data['val'], val_size = None, 0
                if self.val_set is True:
                    train_vids, val_vids = self.split_validation(
                        mode=mode, data=train_vids, groups=train_vid_idx,
                        random_state=0, **inner_kwargs)

                    self.data['val'], val_size = _loadFile(
                        os.path.join(self.dataset_dir, self.files['train']),
                        self.out_layer, val_vids, test_objs)

                self.data['train'], train_size = _loadFile(
                    os.path.join(self.dataset_dir, self.files['train']),
                    self.out_layer, train_vids, test_objs)

                yield (_merge_datasets((self.data['train'], self.data['aloi'])),
                       self.data['val']), {'train': train_size, 'val': val_size}


    def split_validation(self,
                         mode='frame',
                         ratio=None,
                         data=None,
                         labels=None,
                         random_state=None,
                         groups=None,
                         **kwargs):
        """[summary]

        Keyword Arguments:
            mode {str} -- [description] (default: {'frame'})
            ratio {[type]} -- [description] (default: {None})
            data {[type]} -- [description] (default: {None})
            labels {[type]} -- [description] (default: {None})
            random_state {[type]} -- [description] (default: {None})
            groups {[type]} -- [description] (default: {None})

        Raises:
            ValueError -- [description]

        Returns:
            [type] -- [description]
        """

        extra_data = None
        if ratio is None:
            ratio = self.val_ratio

        if mode == 'frame':
            if data is None:
                data, labels = self.data['train']
                extra_data = self.data['aloi']

            x_train, x_val, y_train, y_val = train_test_split(
                data, labels, stratify=labels, test_size=ratio, random_state=random_state)
            train_data = x_train, y_train
            return _merge_datasets((train_data, extra_data)), (x_val, y_val)

        if mode == 'video':
            if data is None:
                data = np.asarray(self.train_entries)

            x_train, x_val, _, _ = train_test_split(
                data, np.ones(len(data)), test_size=ratio, random_state=random_state)

            if labels is None:
                return x_train, x_val

        try:
            raise NotImplementedError
            if data is None:
                data = np.asarray(self.train_entries)

            split_method = getattr(sklearn.model_selection, mode)
            split_method = split_method(**kwargs)
            train_index, val_index = next(
                split_method.split(data, data, groups))
            return data[train_index], data[val_index]

        except AttributeError:
            raise ValueError('invalid mode \'{}\' chosen'.format(mode))


def _merge_datasets(datasets):
    """ Merge (concatenates along samples dimension) together datasets
    (data, label).

    Arguments:
        datasets {list} -- Contains (data, label) elements to be merged together

    Returns:
        {tuple} A single dataset tuple (data, label) consisting of all input datasets
        in the same exact order.
    """
    return tuple(np.concatenate(data) for data in zip(*tuple(filter(None, datasets))))


def _loadFile(basepath, out_layer, vid_name_iter, exceptions=[], verbose=1):
    """Loads data from within the HDF5 file specified by 'basepath' relative
    to the feature maps of the videos 'vid_name_iter' obtained from layer
    'out_layer'. It's possible to consider forbidden substrigs defined by
    'exceptions' such that any data whose path contains such substrings is not
    loaded.

    Arguments:
        basepath {string} -- [description]
        out_layer {string} -- [description]
        vid_name_iter {iterable} -- [description]

    Keyword Arguments:
        exceptions {list} -- [description] (default: {[]})
        verbose {int} -- [description] (default: {1})

    Raises:
        NameError -- [description]
        exception -- [description]

    Returns:
        nd.array -- data
        nd.array -- labels
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

    set_size = [vid_labels.shape[0] for vid_labels in labels]
    data = np.concatenate(data)
    labels = np.concatenate(labels).astype(int)

    h5_file.close()

    return (data, labels), set_size

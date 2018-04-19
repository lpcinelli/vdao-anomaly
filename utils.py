import json
import pickle

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def save_data(data, filename, json_format=True, pickle_format=True):
    if json_format:
        with open(filename + '.json', 'w') as fp:
            json.dump(data, fp, sort_keys=True, indent=4, cls=NumpyEncoder)
    if pickle_format:
        with open(filename + '.pkl', 'wb') as fp:
            pickle.dump(data, fp, protocol=pickle.HIGHEST_PROTOCOL)

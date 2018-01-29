import os
import json
import fnmatch
import numpy as np

# Functions and classes for saving and loading modelspecs


# This next class from:
# https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):  # and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def _modelspec_filename(basepath, number):
    suffix = '.{:04d}.json'.format(number)
    return (basepath + suffix)


def save_modelspec(modelspec, filepath):
    '''
    Saves a modelspec to filepath. Overwrites any existing file.
    '''
    with open(filepath, mode='w+') as f:
        json.dump(modelspec, f, cls=NumpyAwareJSONEncoder)


def save_modelspecs(directory, basename, modelspecs):
    '''
    Saves one or more modelspecs to disk with stereotyped filenames:
        directory/basename.0000.json
        directory/basename.0001.json
        directory/basename.0002.json
        ...etc...
    '''
    basepath = os.path.join(directory, basename)
    for idx, modelspec in enumerate(modelspecs):
        filepath = _modelspec_filename(basepath, idx)
        save_modelspec(modelspec, filepath)


def load_modelspec(filepath):
    '''
    Returns a single modelspecs loaded from filepath.
    '''
    ms = json.load(filepath)
    return ms


def load_modelspecs(directory, basename):
    '''
    Returns a list of modelspecs loaded from directory/basename.*.json
    '''
    regex = '^' + basename + '\.{\d+}\.json'
    files = fnmatch.filter(os.listdir(basename), regex)
    modelspecs = [json.load(f) for f in files]
    return modelspecs

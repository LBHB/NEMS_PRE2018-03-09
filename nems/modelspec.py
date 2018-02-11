import os
import copy
import json
import fnmatch
import numpy as np

from importlib import import_module
from nems.utils import split_to_api_and_fn

# Functions for saving, loading, and evaluating modelspecs


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


lookup_table = {}  # TODO: Replace with real memoization/joblib later


def _lookup_fn_at(fn_path):
    '''
    Private function that returns a function handle found at a
    given module. Basically, a way to import a single function.
    e.g.
        myfn = _lookup_fn_at('nems.modules.fir.fir_filter')
        myfn(data)
        ...
    '''
    if fn_path in lookup_table:
        fn = lookup_table[fn_path]
    else:
        api, fn_name = split_to_api_and_fn(fn_path)
        api_obj = import_module(api)
        fn = getattr(api_obj, fn_name)
        lookup_table[fn_path] = fn
    return fn

def evaluate(rec, modelspec, stop=-1):
    '''
    Given a recording object and a modelspec, return a prediction.
    Does not alter its arguments in any way.
    '''
    # d = copy.deepcopy(rec)  # Paranoid, but 100% safe
    d = copy.copy(rec)  # About 10x faster & fine if Signals are immutable
    for m in modelspec[:stop]:
        fn = _lookup_fn_at(m['fn'])
        kwargs = {**m['fn_kwargs'], **m['phi']}  # Merges both dicts
        new_signals = fn(rec=d, **kwargs)
        if type(new_signals) is not list:
            raise ValueError('Fn did not return list of signals: {}'.format(m))
        for s in new_signals:
            d.add_signal(s)
    return d

# TODO:
# 1. What about collisions between phi and fn_kwargs?
# 2. Error checking. Is anything like this needed?
#    if not (rec and i and o and base and amplitude and shift and kappa):
#        raise ValueError('Not all arguments given to double_exponential')
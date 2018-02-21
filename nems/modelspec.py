import os
import copy
import json
import fnmatch
import numpy as np
import importlib

import nems.utils
from nems.distributions.distribution import Distribution

# Functions for saving, loading, and evaluating modelspecs


# This next class from:
# https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
class NumpyAwareJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if issubclass(type(obj), Distribution):
            return obj.tolist()
        if isinstance(obj, np.ndarray):  # and obj.ndim == 1:
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def get_modelspec_metadata(modelspec):
    '''
    Returns a dict of the metadata for this modelspec.
    '''
    # TODO: Consider putting this elsewhere?
    return modelspec[0].get('meta', {})


def set_modelspec_metadata(modelspec, key, value):
    '''
    Sets a key/value pair in the modelspec's metadata. Purely by convention,
    metadata info for the entire modelspec is stored in the first module.
    '''
    if not modelspec[0].get('meta'):
        modelspec[0]['meta'] = {}
    modelspec[0]['meta'][key] = value
    return modelspec

def get_modelspec_name(modelspec):
    '''
    Returns a string that names this modelspec. Suitable for plotting. 
    '''
    meta = get_modelspec_metadata(modelspec)
    if 'name' in meta:
        return meta['name']

    recording_name = meta.get('recording', 'unknown_recording')
    keyword_string = '_'.join([m['id'] for m in modelspec])
    fitter_name = meta.get('fitter', 'unknown_fitter')
    date = nems.utils.iso8601_datestring()
    guess = '.'.join([recording_name, keyword_string, fitter_name, date])
    return guess

def get_modelspec_longname(modelspec):
    '''
    Returns a LONG name for this modelspec suitable for use in saving to disk.
    '''
    meta = get_modelspec_metadata(modelspec)
    recording_name = meta.get('recording', 'unknown_recording')
    keyword_string = '_'.join([m['id'] for m in modelspec])
    fitter_name = meta.get('fitter', 'unknown_fitter')
    date = nems.utils.iso8601_datestring()
    guess = '.'.join([recording_name, keyword_string, fitter_name, date])
    return guess


def _modelspec_filename(basepath, number):
    suffix = '.{:04d}.json'.format(number)
    return (basepath + suffix)


def save_modelspec(modelspec, filepath):
    '''
    Saves a modelspec to filepath. Overwrites any existing file.
    '''
    with open(filepath, mode='w+') as f:
        json.dump(modelspec, f, cls=NumpyAwareJSONEncoder)


def save_modelspecs(directory, modelspecs, basename=None):
    '''
    Saves one or more modelspecs to disk with stereotyped filenames:
        directory/basename.0000.json
        directory/basename.0001.json
        directory/basename.0002.json
        ...etc...
    Basename will be automatically generated if not provided.
    '''
    for idx, modelspec in enumerate(modelspecs):
        if not basename:
            bname = get_modelspec_longname(modelspec)
        else:
            bname = basename
        basepath = os.path.join(directory, bname)
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
    #regex = '^' + basename + '\.{\d+}\.json'
    # TODO: fnmatch is not matching pattern correctly, replacing
    #       with basic string matching for now.  -jacob 2/17/2018
    #files = fnmatch.filter(os.listdir(directory), regex)
    #       Also fnmatch was returning list of strings? But
    #       json.load expecting file object
    #modelspecs = [json.load(f) for f in files]
    dir_list = os.listdir(directory)
    files = [os.path.join(directory, s) for s in dir_list if basename in s]
    modelspecs = []
    for file in files:
        with open(file, 'r') as f:
            try:
                m = json.load(f)
            except json.JSONDecodeError as e:
                print("Couldn't load modelspec: {0}"
                      "Error: {1}".format(file, e))
            modelspecs.append(m)
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
        api, fn_name = nems.utils.split_to_api_and_fn(fn_path)
        api_obj = importlib.import_module(api)
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
    for m in modelspec:
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

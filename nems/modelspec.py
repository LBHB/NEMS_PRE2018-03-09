import re
import os
import copy
import json
import importlib
import numpy as np
import nems.utils
import nems.uri

# Functions for saving, loading, and evaluating modelspecs

# TODO: In retrospect, this should have been a class, just like Recording.
#       Refactoring would not be too hard and would shorten many of these
#       function names. If you do so, see /docs/planning/models.py and
#       bring the ideas into this file, then delete it from docs/planning.


def get_modelspec_metadata(modelspec):
    '''
    Returns a dict of the metadata for this modelspec. Purely by convention,
    metadata info for the entire modelspec is stored in the first module.
    '''
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


def get_modelspec_shortname(modelspec):
    '''
    Returns a string that is just the module ids in this modelspec.
    '''
    keyword_string = '_'.join([m['id'] for m in modelspec])
    return keyword_string


def get_modelspec_longname(modelspec):
    '''
    Returns a LONG name for this modelspec suitable for use in saving to disk
    without a path.
    '''
    meta = get_modelspec_metadata(modelspec)
    recording_name = meta.get('recording', 'unknown_recording')
    keyword_string = get_modelspec_shortname(modelspec)
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
    nems.uri.save_resource(filepath, json=modelspec)


def save_modelspecs(directory, modelspecs, basename=None):
    '''
    Saves one or more modelspecs to disk with stereotyped filenames:
        directory/basename.0000.json
        directory/basename.0001.json
        directory/basename.0002.json
        ...etc...
    Basename will be automatically generated if not provided.
    '''
    if not os.path.isdir(directory):
        os.makedirs(directory)
        os.chmod(directory, 0o777)

    for idx, modelspec in enumerate(modelspecs):
        if not basename:
            bname = get_modelspec_longname(modelspec)
        else:
            bname = basename
        basepath = os.path.join(directory, bname)
        filepath = _modelspec_filename(basepath, idx)
        save_modelspec(modelspec, filepath)
    return filepath


def load_modelspec(filepath):
    '''
    Returns a single modelspecs loaded from filepath.
    '''
    json_data = open(filepath).read()
    ms = json.loads(json_data)
    return ms


def load_modelspecs(directory, basename, regex=None):
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
    if regex:
        # TODO: Not sure why this isn't working? No errors but
        #       still isn't matching the things it should be matching.
        #       ( tested w/ regex='^TAR010c-18-1\.{\d+}\.json')
        #       -jacob 2/25/18
        if isinstance(regex, str):
            regex = re.compile(regex)
        files = [os.path.join(directory, s) for s in dir_list
                 if re.search(regex, s)]
    else:
        files = [os.path.join(directory, s) for s in dir_list
                 if (basename in s and '.json' in s)]
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


def evaluate(rec, modelspec, stop=None):
    '''
    Given a recording object and a modelspec, return a prediction.
    Does not alter its arguments in any way.
    If stop is none, will use entire list. Otherwise, will only evaluate
    modules 0 through stop-1.
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


def summary_stats(modelspecs):
    '''
    Generates summary statistics for a list of modelspecs.
    Each modelspec must be of the same length and contain the same
    modules (though they need not be in the same order).

    For example, ten modelspecs composed of the same modules that
    were fit to ten different datasets can be compared. However, ten
    modelspecs all with different modules fit to the same data cannot
    be compared because there is no guarantee that they contain
    comparable parameter values.

    Arguments:
    ----------
    modelspecs : list of modelspecs
        See docs/modelspecs.md

    Returns:
    --------
    means, stds : dicts
        Each contains one key for each parameter, of the form:
            {'<modelspec_index>_<parameter_name>': <mean value>}
            or {'<modelspec_index>_<parameter_name>': <standard deviation>}
    '''
    # Don't modify the modelspecs themselves
    modelspecs = [m.copy() for m in modelspecs]

    # Modelspecs must have the same length to compare
    # TODO: Remove this requirement? Would just need some handling of
    #       missing indices in the rest of the function's logic.
    #       Keeping the requirement lets us make some simplifying assumptions.
    length = None
    for m in modelspecs:
        if length:
            if len(m) != length:
                raise ValueError("All modelspecs must have the same length")
        length = len(m)

    # Modelspecs must have the same modules to compare
    # TODO: Remove this requirement? Same issue as with length matching.
    fns = [m['fn'] for m in modelspecs[0]]
    for mspec in modelspecs[1:]:
        m_fns = [m['fn'] for m in mspec]
        if not sorted(fns) == sorted(m_fns):
            raise ValueError("All modelspecs must have the same modules")

    # Assumble a dict of columns for creating a Dataframe, with
    # the column name format: <modelspec_index>_<parameter>
    columns = {}
    for i, m in enumerate(modelspecs[0]):
        params = m['phi'].keys()
        for p in params:
            columns.update({'{0}_{1}'.format(i, p): []})

    for col in columns.keys():
        # First chunk, before _, is the 'module' index within modelspec
        split = col.split('_')
        m = int(split[0])
        # Second chunk, after _, is the parameter
        p = '_'.join(split[1:])
        for mspec in modelspecs:
            this_p = mspec[m]['phi'][p]
            columns[col].append(this_p)

    # Now columns should look something like:
    # {'0_mu': [1, 1, 3, 4],
    #  '0_sd': [0.5, 1, 0.5, 1],
    #  '1_kappa': [1.0, 3.0, 2.0, 4.0]}

    # TODO: Currently gets a single scalar mean/std for parameters like
    #       weight_channels' coefficients. Might want to end up with
    #       an array of mean/stds for those instead?
    means = columns.copy()
    stds = columns.copy()
    for col, values in columns.items():
        means[col] = np.mean(values)
        stds[col] = np.std(values)

    # TODO: Might be better to have this end up as a single dictionary
    #       of the form:
    #       {'<fn entry>_<param name>': {'mean': x, 'std': y}}
    #       ex:
    #       {'nems.modules.nonlinearity.dexp_kappa': {'mean': 1.0,
    #                                                 'std': 0.37}}

    return means, stds

# TODO: Check that the word 'phi' is not used in fn_kwargs
# TODO: Error checking the modelspec before execution;
# TODO: Validation of modules json schema; all require args should be present

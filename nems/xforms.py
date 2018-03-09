import io
import copy
import logging
import socket
import nems.analysis.api
import nems.initializers as init
import nems.metrics as metrics
import nems.modelspec as ms
from nems.modelspec import set_modelspec_metadata, get_modelspec_metadata, get_modelspec_shortname
import nems.plots.api as nplt
import nems.preprocessing as preproc
import nems.priors as priors
from nems.uri import save_resource, load_resource
from nems.utils import iso8601_datestring
from nems.fitters.api import scipy_minimize
from nems.recording import Recording

log = logging.getLogger(__name__)

xforms = {}  # A mapping of kform keywords to xform 2-tuplets (2 element lists)


def defxf(keyword, xformspec):
    '''
    Adds xformspec to the xforms keyword dictionary.
    A helper function so not every keyword mapping has to be in a single
    file and part of a very large single multiline dict.
    '''
    if keyword in xforms:
        raise ValueError("Keyword already defined! Choose another name.")
    xforms[keyword] = xformspec


def load_xform(uri):
    '''
    Loads and returns xform saved as a JSON.
    '''
    xform = load_resource(uri)
    return xform


def xfspec_shortname(xformspec):
    '''
    Given an xformspec, makes a shortname for it.
    '''
    n = len('nems.xforms.')
    fn_names = [xf[n:] for xf, xfa in xformspec]
    name = ".".join(fn_names)
    return name


def evaluate(xformspec, context={}, stop=None):
    '''
    Similar to modelspec.evaluate, but for xformspecs, which is a list of
    2-element lists of function and keyword arguments dict. Each XFORM must
    return a dictionary of the explicit changes made to the context, which
    is the dict that is passed from xform to xform.

    Also, this function wraps every logging call and saves it in a log
    that is the second value returned by this function.
    '''
    context = copy.deepcopy(context)  # Create a new starting context

    # Create a log stream set to the debug level; add it as a root log handler
    log_stream = io.StringIO()
    ch = logging.StreamHandler(log_stream)
    ch.setLevel(logging.DEBUG)
    fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(fmt)
    ch.setFormatter(formatter)
    rootlogger = logging.getLogger()
    rootlogger.addHandler(ch)

    # Evaluate the xforms
    for xfa in xformspec[:stop]:
        if len(xfa) != 2:
            raise ValueError('got non 2-tuple for xform: {}'.format(xfa))
        xf = xfa[0]
        xfargs = xfa[1]
        fn = ms._lookup_fn_at(xf)
        # Check for collisions; more to avoid confusion than for correctness:
        for k in xfargs:
            if k in context:
                m = 'xf arg {} overlaps with context: {}'.format(k, xf)
                raise ValueError(m)
        # Merge args into context, and make a deepcopy so that mutation
        # inside xforms will not be propogated unless the arg is returned.
        merged_args = {**xfargs, **context}
        args = copy.deepcopy(merged_args)
        # Run the xf
        log.info('Evaluating: {}({})'.format(xf, args))
        new_context = fn(**args)
        # Use the new context for the next step
        if type(new_context) is not dict:
            raise ValueError('xf did not return a context dict: {}'.format(xf))
        context = {**context, **new_context}

    # Close the log, remove the handler, and add the 'log' string to context
    log.info('Done (re-)evaluating xforms.')
    ch.close()
    rootlogger.removeFilter(ch)

    return context, log_stream.getvalue()


###############################################################################
# Stuff below this line are useful resuable components.
# See xforms_test.py for how to use it.


def load_recordings(recording_uri_list, **context):
    '''
    Load one or more recordings into memory given a list of URIs.
    '''
    rec = Recording.load(recording_uri_list[0])
    other_recordings = [Recording.load(uri) for uri in recording_uri_list[1:]]
    if other_recordings:
        rec.concatenate_recordings(other_recordings)
    return {'rec': rec}


def add_average_sig(rec, signal_to_average, new_signalname, epoch_regex,
                    **context):
    rec = preproc.add_average_sig(rec,
                                  signal_to_average=signal_to_average,
                                  new_signalname=new_signalname,
                                  epoch_regex=epoch_regex)
    return {'rec': rec}


def split_by_occurrence_counts(rec, **context):
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    return {'est': est, 'val': val}


def average_away_stim_occurrences(est, val, **context):
    est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    return {'est': est, 'val': val}

def average_away_stim_occurrences_rec(rec, **context):
    rec = preproc.average_away_epoch_occurrences(rec, epoch_regex='^STIM_')
    return {'rec': rec}


def split_at_time(rec, fraction, **context):
    est, val = rec.split_at_time(fraction)
    return {'est': est, 'val': val}


def use_all_data_for_est_and_val(rec, **context):
    est = rec
    val = rec
    return {'est': est, 'val': val}


def init_from_keywords(keywordstring, IsReload=False, **context):
    if not IsReload:
        modelspec = init.from_keywords(keywordstring)
        return {'modelspecs': [modelspec]}
    else:
        return {}


def load_modelspecs(modelspecs, uris,
                    IsReload=False, **context):
    '''
    i.e. Load a modelspec from a specific place. This is not
    the same as reloading a model for later inspection; it would be more
    appropriate when doing something complicated with several different
    models.
    '''
    if not IsReload:
        modelspecs = [load_resource(uri) for uri in uris]
    return {'modelspecs': modelspecs}


def set_random_phi(modelspecs, IsReload=False, **context):
    ''' Starts all modelspecs at random phi sampled from the priors. '''
    if not IsReload:
        modelspecs = [priors.set_random_phi(m) for m in modelspecs]
    return {'modelspecs': modelspecs}


def fit_basic_init(modelspecs, est, IsReload=False, **context):
    ''' A basic fit that optimizes every input modelspec. '''
    if not IsReload:
        modelspecs = [nems.initializers.prefit_to_target(
                est, modelspec, nems.analysis.api.fit_basic, 'levelshift',
                fitter=scipy_minimize,
                fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}}) 
                for modelspec in modelspecs]
    return {'modelspecs': modelspecs}

def fit_basic(modelspecs, est, IsReload=False, **context):
    ''' A basic fit that optimizes every input modelspec. '''
    if not IsReload:
        modelspecs = [nems.analysis.api.fit_basic(est,
                                                  modelspec,
                                                  fitter=scipy_minimize)[0]
                      for modelspec in modelspecs]
    return {'modelspecs': modelspecs}


def fit_n_times_from_random_starts(modelspecs, est, ntimes,
                                   IsReload=False, **context):
    ''' Self explanatory. '''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = [nems.analysis.api.fit_from_priors(est,
                                                        modelspec,
                                                        ntimes=ntimes)
                      for modelspec in modelspecs]
    return {'modelspecs': modelspecs}


def fit_random_subsets(modelspecs, est, nsplits,
                       IsReload=False, **context):
    ''' Randomly sample parts of the data? Wait, HOW DOES THIS WORK? TODO?'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = nems.analysis.api.fit_random_subsets(est,
                                                          modelspecs[0],
                                                          nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_equal_subsets(modelspecs, est, nsplits,
                      IsReload=False, **context):
    ''' Divide the data into nsplits equal pieces and fit each one.'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = nems.analysis.api.fit_subsets(est,
                                                   modelspecs[0],
                                                   nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_jackknifes(modelspecs, est, njacks,
                   IsReload=False, **context):
    ''' Jackknife the data, fit on those, and make predictions from those.'''
    if not IsReload:
        if len(modelspecs) > 1:
            raise ValueError('I only work on 1 modelspec')
        modelspecs = nems.analysis.api.fit_jackknifes(est,
                                                      modelspecs[0],
                                                      njacks=njacks)
    return {'modelspecs': modelspecs}


def save_recordings(modelspecs, est, val, **context):
    # TODO: Save the recordings somehow?
    return {'modelspecs': modelspecs}


def add_summary_statistics(modelspecs, est, val, **context):
    # modelspecs = metrics.add_summary_statistics(est, val, modelspecs)
    # TODO: Add statistics to metadata of every modelspec
    
    modelspecs=nems.analysis.api.standard_correlation(est,val,modelspecs)
    
    return {'modelspecs': modelspecs}


def plot_summary(modelspecs, val, figures=None, **context):
    # CANNOT initialize figures=[] in optional args our you will create a bug
    if not figures:
        figures = []
    fig = nplt.plot_summary(val, modelspecs)
    # Needed to make into a Bytes because you can't deepcopy figures!
    figures.append(nplt.fig2BytesIO(fig))
    return {'figures': figures}


def fill_in_default_metadata(rec, modelspecs, IsReload=False, **context):
    '''
    Sets any uninitialized metadata to defaults that should help us
    find it in nems_db again. (fitter, recording, date, etc)
    '''
    if not IsReload:
        # Add metadata to help you reload this state later
        for modelspec in modelspecs:
            meta = get_modelspec_metadata(modelspec)
            if 'fitter' not in meta:
                set_modelspec_metadata(modelspec, 'fitter', 'None')
            if 'fit_time' not in meta:
                set_modelspec_metadata(modelspec, 'fitter', 'None')
            if 'recording' not in meta:
                recname = rec.name if rec else 'None'
                set_modelspec_metadata(modelspec, 'recording', recname)
            if 'recording_uri' not in meta:
                uri = rec.uri if rec and rec.uri else 'None'
                set_modelspec_metadata(modelspec, 'recording_uri', uri)
            if 'date' not in meta:
                set_modelspec_metadata(modelspec, 'date', iso8601_datestring())
            if 'hostname' not in meta:
                set_modelspec_metadata(modelspec, 'hostname',
                                       socket.gethostname())
    return {'modelspecs': modelspecs}


# TODO: Perturb around the modelspec to get confidence intervals

# TODO: Use simulated annealing (Slow, arguably gets stuck less often)
# modelspecs = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.annealing)

# TODO: Use Metropolis algorithm (Very slow, gives confidence interval)
# modelspecs = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.metropolis)

# TODO: Use 10-fold cross-validated evaluation
# fitter = partial(nems.cross_validator.cross_validate_wrapper, gradient_descent, 10)
# modelspecs = nems.analysis.fit_cv(est, modelspec, folds=10)


def tree_path(recording, modelspecs, xfspec):
    '''
    Returns a relative path (excluding filename, host, port) for URIs.
    Editing this function edits the path in the file tree of every
    file saved!
    '''

    xformname = xfspec_shortname(xfspec)
    modelname = get_modelspec_shortname(modelspecs[0])
    recname = recording.name  # Or from rec.uri???
    meta = get_modelspec_metadata(modelspecs[0])
    date = meta.get('date', iso8601_datestring())

    path = '/' + recname + '/' + modelname + '/' + xformname + '/' + date + '/'

    return path


def save_analysis(destination,
                  recording,
                  modelspecs,
                  xfspec,
                  figures,
                  log):
    '''Save an analysis file collection to a particular destination.'''

    treepath = tree_path(recording, modelspecs, xfspec)
    destination = destination[:-1] if destination[-1] == '/' else destination
    base_uri = destination + treepath

    xfspec_uri = base_uri + 'xfspec.json'  # For attaching to modelspecs

    for number, modelspec in enumerate(modelspecs):
        set_modelspec_metadata(modelspec, 'xfspec', xfspec_uri)
        save_resource(base_uri + 'modelspec.{:04d}.json'.format(number),
                      json=modelspec)
    for number, figure in enumerate(figures):
        save_resource(base_uri + 'figure.{:04d}.png'.format(number),
                      data=figure)
    save_resource(base_uri + 'log.txt', data=log)
    save_resource(xfspec_uri, json=xfspec)
    return {}

import io
import copy
import logging
import nems.analysis.api
import nems.initializers as init
import nems.metrics as metrics
import nems.priors as priors
import nems.modelspec as ms
import nems.preprocessing as preproc
import nems.urls as urls
import nems.plots as nplt

from nems.fitters.api import scipy_minimize
from nems.recording import Recording
from nems.keywords import _lookup_fn_at

log = logging.getLogger(__name__)

xforms = {}  # A mapping of kform keywords to xform functions


def defxf(keyword, xformspec):
    '''
    Adds xformspec to the xforms keyword dictionary.
    A helper function so not every keyword mapping has to be in a single
    file and part of a very large single multiline dict.
    '''
    if keyword in xforms:
        raise ValueError("Keyword already defined! Choose another name.")
    xforms[keyword] = xformspec


def evaluate(context, xformspec, stop=None):
    '''
    Just like modelspec.evaluate, but for xformspecs.
    Wraps everything with a log file context, as well.
    '''
    context = copy.deepcopy(context)  # Create a new starting context

    # Create a logging stream at the debug level, and add it as a handler
    log_stream = io.StringIO()
    ch = logging.StreamHandler(log_stream)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # ch.setFormatter(formatter)
    log.addHandler(ch)

    # Evaluate the xforms
    for xf in xformspec[:stop]:
        fn = _lookup_fn_at(xf['xf'])
        # Check for collisions; more to avoid confusion than for correctness:
        for k in xf['xf_kwargs']:
            m = 'xf kwarg {} overlaps with context: {}'.format(k, xf)
            raise ValueError(m)
        # Merge args into context
        args = {**xf['xf_kwargs'], **xf['context']}
        # Run the xf
        log.info('Evaluating: {}'.xf['xf'])
        new_context = fn(**args)
        # Use the new context for the next step
        if type(new_context) is not dict:
            raise ValueError('xf did not return a context dict: {}'.format(xf))
        context = new_context
    # Close the log, remove the handler, and add the 'log' string to context
    ch.close()
    log.removeFilter(ch)
    context['log'] = log_stream.get_value()
    return context


def load_recordings(recording_uri_list, **context):
    '''
    Load one or more recordings into memory given a list of URIs.
    '''
    rec = Recording.load(recording_uri_list[0])
    other_recordings = [Recording.load(uri) for uri in recording_uri_list[1:]]
    rec.concatenate_recordings(other_recordings)
    return {'rec': rec}


def add_respavg(rec, **context):
    rec = preproc.add_average_sig(rec,
                                  signal_to_average='resp',
                                  new_signalname='resp',
                                  epoch_regex='^STIM_')
    return {'rec': rec}


def split_est_val_by_stim_repetition_count(rec, **context):
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
    return {'est': est, 'val': val}


def average_away_stim_occurrences(est, val, **context):
    est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    return {'est': est, 'val': val}


def split_at_time(rec, fraction, **context):
    est, val = rec.split_at_time(fraction)
    return {'est': est, 'val': val}


def use_all_data_for_est_and_val(rec, **context):
    est = rec
    val = rec
    return {'est': est, 'val': val}


# 'wc18x1_lvl1_fir15x1_dexp1'
def initialize_modelspec(keywordstring, **context):
    modelspec = init.from_keywords(keywordstring)
    return {'modelspecs': [modelspec]}


def load_modelspecs(uris, **context):
    modelspecs = [ms.load_modelspecs(uri) for uri in uris]
    return {'modelspecs': modelspecs}


def start_random_phi(modelspecs, **context):
    ''' Starts all modelspecs at random phi sampled from the priors. '''
    modelspecs = [priors.set_random_phi(m) for m in modelspecs]
    return {'modelspecs': modelspecs}


def fit_basic(modelspecs, est, **context):
    ''' A basic fit that returns a single modelspec. '''
    modelspecs = [nems.analysis.api.fit_basic(est,
                                              modelspec,
                                              fitter=scipy_minimize)
                  for modelspec in modelspecs]
    return {'modelspecs': modelspecs}


def fit_n_times_from_random_starts(modelspecs, est, ntimes, **context):
    ''' Self explanatory. '''
    if len(modelspecs) > 1:
        raise ValueError('I only work on 1 modelspec')
    modelspecs = [nems.analysis.api.fit_from_priors(est,
                                                    modelspec[0],
                                                    ntimes=ntimes)
                  for modelspec in modelspecs]
    return {'modelspecs': modelspecs}


def fit_random_subsets(modelspecs, est, nsplits, **context):
    ''' Randomly sample parts of the data? Wait, HOW DOES THIS WORK? TODO?'''
    if len(modelspecs) > 1:
        raise ValueError('I only work on 1 modelspec')
    modelspecs = nems.analysis.api.fit_random_subsets(est,
                                                      modelspecs[0],
                                                      nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_equal_subsets(modelspecs, est, nsplits, **context):
    ''' Divide the data into nsplits equal pieces and fit each one.'''
    if len(modelspecs) > 1:
        raise ValueError('I only work on 1 modelspec')
    modelspecs = nems.analysis.api.fit_subsets(est,
                                               modelspec,
                                               nsplits=nsplits)
    return {'modelspecs': modelspecs}


def fit_jackknifes(modelspecs, est, njacks, **context):
    ''' Jackknife the data, fit on those, and make predictions from those.'''
    if len(modelspecs) > 1:
        raise ValueError('I only work on 1 modelspec')
    modelspecs = nems.analysis.api.fit_jackknifes(est,
                                                  modelspec,
                                                  njacks=njacks)
    return {'modelspecs': modelspecs}


def save_it_all(modelspecs, est, val, log, results, base_destination,
                **context):
    ''' Default save routine. '''
    relative_path = urls.relative_path(modelspecs[0])
    destination = base_destination + relative_path
    # log.info('Computing performance...')
    modelspecs = metrics.add_summary_statistics(est, val, modelspecs)
    # log.info('Saving Modelspecs...')
    ms.save_modelspecs(destination, modelspecs)
    # log.info('Plotting...')
    figures = []
    figures[0] = nplt.plot_summary(val, modelspecs)
    figures[1] = nplt.plot_models(val, modelspecs)
    # log.info('Saving plots...')
    # save_plots(base_destination, figures)
    # log.info('Saving logfile...')
    # save_logfile(destination, log)



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

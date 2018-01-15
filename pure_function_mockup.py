# Stateless Module Demo

# The more I think about it, the more I wonder if our real goal should just
# be to exorcise all state from a module, turning modules into "pure functions"
# (a.k.a. functions without side effects. Not that we shouldn't use classes,
# but that we should keep focused on stateless functions when possible.
#
# Managing modules can be complicated precisely because they contain mutable
# state. Given that state is usually easier when it is all in once place,
# maybe packing the entire model into a single data structure isn't such a
# crazy idea.
#
# The following shows a little demo of how that might look in general,
# and for three cases that are not supported by the current version of NEMS:
#
#   1. "Per-module fitters", when each module uses a different sub-fitter
#      and all modules are iterated through.
#
#   2. "Parameter fitting dropout", when some parameters are randomly NOT
#      considered for optimization during the fitting process
#
#   3. "Module dropout", when the transformation of random modules are
#      temporarily omitted during the fitting process
#
# The latter two techniques have been used extensively in deep learning
# and allegedly make the fitting process more robust. And per-module
# fitters could be a cleaner way of putting all the fitting code in a single
# place rather than spreading it out and putting a little in each module,
# which then requires that other fitters remove manually whenever they
# do not want to do an "initial-fit".

##############################################################################

# ----------------------------------------------------------------------------
# DATA LOADING AND SPLITTING

import nems.signal as signal

# Load the data: a dict of Signal objects
sigs = signal.load_signals_in_dir('signals/gus027b13_p_PPS/')

# Alternatively, from the (soon to be created) API:
URL = "neuralprediction.org:3003/signals?batch=273&cellid=gus027b13-a1"
sigs = fetch_signals_over_http(URL)

# For now, just assume that sigs looks roughly like this:
# sigs = {'stim': signal.Signal(...),
#         'resp': signal.Signal(...),
#         'pupil': signal.Signal(...)}

# Create est dataset from the first 80%, and val from the last 20%
# (The Signal class has other ways of splitting/combining files & signals)
est, val = signal.split_signals_by_time(sigs, 0.8)

# Open question: even though it is only a few lines, how and where should
# this information be recorded? The data set that a model was trained on
# is relevant information that should be serialized and recorded somewhere.
# save_to_disk('/some/path/model1_dataspec.json',
#              json.dumps({'URL': URL, 'est_val_split_frac': 0.8}))

# ----------------------------------------------------------------------------
# MODELSPEC
# A 'modelspec' object is a simple datastructure that defines the entire
# model, and is easily saved/loaded to disk. It is essentially a serialization
# of the current "stack" object. You can define it three ways:

# Modelspec Creation Method #1: specify it manually
modelspec = [{'fn': 'nems.modules.fir',       # The pure function to call
              'phi': [1.0, 0.9, 20.8, ...],   # current params during fitting
              'prior': [0.0, 0.0, 0.1, ...],  # a.k.a. Initialization params
              'posterior': None,              # a.k.a. Params after fitting
              'plotfn': 'nems.plots.plot_strf'}, # A plotting function
             {'fn': 'nems.modules.nl/dexp',
              'phi': [0.54, 1.982, 9.1],
              'prior': [1, 2, 5],
              'posterior': None,
              'plotfn': 'nems.plots.plot_nl'},
             ...]

# Modelspec Creation Method #2: load a from disk
# modelspec = json.loads('/path/to/model1.json')

# Modelspec Creation Method #3: create from "shorthand/default" keyword string
# modelspec = keywords_to_modelspec('fir30_dexp')

# ----------------------------------------------------------------------------
# FITTING ALGORITHMS
#
# I'm going to define some words to keep names straight:
#
# | Function Signature                   | NAME. Description                     |
# |--------------------------------------+---------------------------------------|
# | f(modelspec, data) -> pred           | EVALUATOR. Makes a model's prediction |
# | f(pred) -> error                     | METRIC. Evaluates predictive accuracy |
# | f(modelspec, evaluator) -> modelspec | FITTER. Finds a better modelspec      |
#
# where:
#   data       is a dict of signals, like {'stim': ..., 'resp': ..., ...}
#   pred       is a dict of signals, just like 'data' but containing 'pred'
#   modelspec  is defined as above

def evaluator(data_in, modelspec):
    ''' This should probably be a put into a StackModel class; but
    basically it is just a way of executing a modelspec on some data.'''
    data = data_in
    for m in modelspec:
        data = m['fn'](m.phi, data)
    return data

metric = lambda data: nems.metrics.MSE(data['resp'], data['pred'])

# Make a curried evaluator function for the fitter (so that it only sees
# the estimation data set, and never the validation one)
from functools import partial
est_evaluator = partial(evaluator, est)
# TODO: If desired, evaluator could include inner CV loop wrapper as well

# Perform the fit on est data set using standard gradient_descent
fitter = nems.fitter.gradient_descent
modelspec_fitted = fitter(modelspec, est_evaluator)

# Take your final measurements on the validation data set, and save!
results = {}
for m in nems.metrics.get_some_list_of_metric_functions:
    results[m.name] = m(val['resp'], evaluator(val, modelspec_fitted))

save_to_disk('/some/path/model1.json', json.dumps(modelspec_fitted))
save_to_disk('/some/path/model1_results.json', json.dumps(results))

# ----------------------------------------------------------------------------
# INTERLUDE: Giving names to fitter internal functions
#
# We are about to get into more difficult territory, and so once again I am
# going to make some mathematical definitions that may be useful as
# optional arguments (or internal functions) used in our fitters.
#
# |---------------------------+------------------------------------------------|
# | f(modelspec) -> error     | COST_FN. (Closed over function references EST) |
# | f(stepinfo) -> boolean    | STOP_COND. Decides when to stop the fit        |
# | f(modelspec) -> stepspace | PACKER. Converts a modelspec into a stepspace  |
# | f(stepspace) -> modelspec | UNPACKER. The inverse operation of PACKER      |
# | f(stepspace) -> stepspace | STEPPER. Alters 1+ parameters in stepspace     |
#
# I used to think that "packers" and "unpackers" should always convert to
# vectors, combining all params into "phi_all". However, now I think it may be
# better to consider them as generating a point in "stepspace" in which the
# fitting algorithm may take a step, and leave the nature of this subspace
# vague -- it may not be a vector space, and the number of dimensions is not
# fixed.
#
# My rationale for this is that:
#
# 1) I can think of certain fitting situations in which the modelspec should not
#    become a parameter vector (such as during a bayesian fit, where the
#    modelspec gives a distribution rather than a single parameter);
#
# 2) We may want to develop different "packers" and "unpackers" that strive to
#    keep the parameter search space (the "stepspace) linear, even if the
#    parameters themselves create a fairly nonlinear space.
#
# 3) By not assuming anything about the stepspace, we can easily change its
#    dimensionality from step to step by creating a triplet of functions on
#    every step that pack, step, and unpack themselves.

# ----------------------------------------------------------------------------
# Example: fitting iteratively, giving every module a different fitter

# Everything is as before, except for the one line that runs the fitter is
# replaced with the following:

import time
import skopt
from nems.termination_condition import error_nondecreasing

per_module_fitters = {'nems.modules.fir': skopt.gp_minimize,
                      'nems.modules.nl/dexp': nems.fitters.take_greedy_10_steps}

def iterative_permodule_fitter(fitterdict, modelspec, cost_fn,
                               stop_cond=error_nondecreasing):
    '''Fit each module, one at a time'''
    spec = modelspec
    stepinfo = {'num': 1, # Num of steps
                'err': cost_fn(spec), #
                'err_delta': None, # Change since last step
                'start_time': time.time()}
    # Check for errors
    for m in modelspec:
        if m['fn'] not in fitterdict:
            raise ValueError('Missing per module fitter:'+m['fn'])
    # Now do the actual iterated fit
    while error_nondecreasing(stepinfo):
        for m in modelspec:
            m['phi'] = fitterdict[m['fn']](m['phi'])
        stepinfo = update_stepinfo() # TODO
    return spec

# Build the fitter fn and do the fitting process
fitter = partial(iterative_permodule_fitter, per_module_fitters)
modelspec_fitted = fitter(modelspec, est_evaluator)


# ----------------------------------------------------------------------------
# Even harder example: fitting different random subsets of the parameters
# on each fitter search round

def random_subset_fitter(modelspec, cost_fn,
                         stop_cond=...):
    cost_fn = lambda spec: metric(evaluator(spec))
    spec = modelspec
    while not stop_cond(stepinfo):
        (packer, stepper, unpacker) = make_random_subset_triplet()  # TODO
        stepspace_point = packer(modelspec)
        best_spec = try_every_step(stepspace_point, unpacker, cost_fn)
        spec = unpacker(best_spec)
    return spec


# ----------------------------------------------------------------------------
# I think you can see how you could have fitters:
# - Return multiple parameter sets, such as in a bayesian analysis or swarm model
# - Completely drop out entire modules or parameters (because the modules are
#      not really in existance; they are just part of the modelspec datastructure)
# - Return the whole set of modelspecs trained on the jackknifed est data
# etc

# A demonstration of a minimalist model fitting system

from nems.recording import Recording

# ----------------------------------------------------------------------------
# DATA WRANGLING

# Method #1: Load the data from a local directory
rec = Recording.load('signals/gus027b13_p_PPS/')

# Method #2: Load the data using the (soon to be created, TODO) HTTP API:
# URL = "neuralprediction.org:3003/signals?batch=273&cellid=gus027b13-a1"
# rec = fetch_signals_over_http(URL)


# ----------------------------------------------------------------------------

# Create estimation dataset from the first 80% of the recording, and the
# validation data set from the last 20%. (Note: the Recording class has other
# ways of splitting/combining things)
est, val = rec.split_at_time(0.8)

# TODO: Open question: even though it is only a few lines, how and where should
# this information be recorded? The data set that a model was trained on
# is relevant information that should be serialized and recorded somewhere.
# save_to_disk('/some/path/model1_dataspec.json',
#              json.dumps({'URL': URL, 'est_val_split_frac': 0.8}))


# ----------------------------------------------------------------------------
# CREATING A MODEL FROM A MODELSPEC

# A 'modelspec' object is a simple datastructure that defines the entire
# model, and is easily saved/loaded to disk. It is essentially a serialization
# of a Model object. You can define it three ways:

# Method #1: create from "shorthand/default" keyword string
# modelspec = keywords_to_modelspec('fir30_dexp')

# Method #2: load a JSON from disk
# modelspec = json.loads('modelspecs/model1.json')

# Method #3: specify it manually
modelspec = [{'fn': 'nems.modules.weight_channels',
              'phi' : TODO}
             {'fn': 'nems.modules.fir',       # The pure function to call
              'phi': {                        # The parameters that may change
                  'mu': {
                    # expected distribution
                    'prior': ('Normal', {'mu': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                         'sd': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}),
                    # fitted distribution (if applicable)
                    'posterior': None,
                    # initial scalar value (typically the mean of the prior)
                    'initial': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    # fitted scalar value (if applicable)
                    'final': None,
                    },
                  'sd': {
                    'prior': ('HalfNormal', {'sd': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}),
                    'posterior': None,
                    'initial': 1,
                    'final': None,
                    }
                  },
              'plotfn': 'nems.plots.plot_strf'}, # A plotting function
             ]

# ----------------------------------------------------------------------------
# FITTING ALGORITHMS
#
# I'm going to define some words to keep names straight:
#
# | Function Signature                   | NAME. Description                     |
# |--------------------------------------+---------------------------------------|
# | f(mspec, data) -> pred               | EVALUATOR. Makes a model's prediction |
# | f(pred) -> error                     | METRIC. Evaluates predictive accuracy |
# | f(mspec, metric, evaluator) -> mspec | FITTER. Finds a better modelspec      |
#
# where:
#   data       is a dict of signals, like {'stim': ..., 'resp': ..., ...}
#   pred       is a dict of signals, just like 'data' but containing 'pred'
#   mspec      is the modelspec data type, as was defined above

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
modelspec_fitted = fitter(modelspec, metric, est_evaluator)

# Take your final measurements on the validation data set, and save!
results = {}
for m in nems.metrics.get_some_list_of_metric_functions:
    results[m.name] = m(val['resp'], evaluator(val, modelspec_fitted))

save_to_disk('/some/path/model1.json', json.dumps(modelspec_fitted))
save_to_disk('/some/path/model1_results.json', json.dumps(results))

# Plot the prediction vs reality
# phi_distributions.plot('/some/other/path.png')

# Plot prediction and confidence intervals
# pred_EV = model.evaluate(phi_distributions.expected_value(), val)
# pred_10 = model.evaluate(phi_distributions.ppf(10), val)
# pred_90 = model.evaluate( phi_distributions.ppf(90), val)
# plot_signals('/some/path.png', pred_EV, pred_10, pred_90, ...)

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
#    keep the parameter search space (the "stepspace") linear, even if the
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
        best_spec = stepper(stepspace_point, unpacker, cost_fn)
        spec = unpacker(best_spec)
    return spec


# ----------------------------------------------------------------------------
# I think you can see how you could have fitters:
# - Return multiple parameter sets, such as in a bayesian analysis or swarm model
# - Completely drop out entire modules or parameters (because the modules are
#      not really in existance; they are just part of the modelspec datastructure)
# - Return the whole set of modelspecs trained on the jackknifed est data
# etc


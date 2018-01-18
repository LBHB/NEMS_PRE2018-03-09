# A demonstration of a minimalist model fitting system

from nems.recording import Recording

# ----------------------------------------------------------------------------
# DATA FETCHING

# GOAL: Get your data loaded into memory

# Method #1: Load the data from a local directory
rec = Recording.load('signals/gus027b13_p_PPS/')

# alternative ways to define the data object that could be saved as a 
# short(!) string in the dataspec
rec = my_create_recording_fun('signals/gus027b13_p_PPS/')
rec = Recording.load_standard_nems_format('signals/gus027b13_p_PPS/')


# Method #2: Load the data from baphy using the (incomplete, TODO) HTTP API:
# URL = "neuralprediction.org:3003/signals?batch=273&cellid=gus027b13-a1"
# rec = fetch_signals_over_http(URL)

# Method #3: Load the data from S3:
# stimfile="https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"+cellid+"_NAT_stim_ozgf_c18_fs100.mat"
# respfile="https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"+cellid+"_NAT_resp_fs100.mat"
# rec = fetch_signals_over_http(stimfile, respfile)

# Method #4: Load the data from a jerb (TODO)

# Method #5: Create a Recording object from a matrix, manually (TODO)


# ----------------------------------------------------------------------------
# DATA WITHHOLDING 

# GOAL: Split your data into estimation and validation sets

# Method #1: Split based on time. (First 80% is estimation, last 20% is validation)
est, val = rec.split_at_time(0.8)

# Method #2: Split based on repetition number, rounded to the nearest rep.
# est, val = rec.split_at_rep(0.8)

# Method #3: Create multiple est and val sets as an outer cross-validation loop
# TODO:

# TODO: Open question: even though it is only a few lines, how and where should
# this information be recorded? The data set that a model was trained on
# is relevant information that should be serialized and recorded somewhere.
# save_to_disk('/some/path/model1_dataspec.json',
#              json.dumps({'URL': URL, 'est_val_split_frac': 0.8}))
# TODO: This annotation should be done automatically when split_at_time is called?

# ----------------------------------------------------------------------------
# DEFINE A MODELSPEC

# GOAL: Uniquely define the structure of the model you wish to fit or use to 
# make a prediction about your data. A 'modelspec' is a datastructure that
# defines the entire model and is easily saved/loaded to disk. 
# It is essentially a serialization of a Model object and may be defined using:

# Method #1: create from "shorthand/default" keyword string
# modelspec = keywords_to_modelspec('fir30_dexp')

# Method #2: load a JSON from disk
# modelspec = json.loads('modelspecs/model1.json')

# Method #3: Load it from a jerb (TODO)
# modelspec = ...

# Method #4: specify it manually
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
# DEFINE THE COST FUNCTION
#
# Goal: Define the cost function and metric for use by the fitter.
#
# To help with clarity, we will define the following words mathematically:
# 
# |-----------+----------------------------------------------------------|
# | Name      | Function Signature and Description                       |
# |-----------+----------------------------------------------------------|
# | EVALUATOR | f(mspec, data) -> pred                                   |
# |           | Makes a prediction based on the model and data.          |
# |-----------+----------------------------------------------------------|
# | METRIC    | f(pred) -> error                                         |
# |           | Evaluates the accuracy of the prediction.                |
# |-----------+----------------------------------------------------------|
# | FITTER    | f(mspec, cost_fn) -> mspec                               |
# |           | Tests various points and finds a better modelspec.       |
# |-----------+----------------------------------------------------------|
# | COST_FN   | f(mspec) -> error                                        |
# |           | A function that gives the cost (error) of each mspec.    |
# |           | Often uses curried EST dataset, METRIC() and EVALUATOR() |
# |-----------+----------------------------------------------------------|
#
# where:
#   data       is a dict of signals, like {'stim': ..., 'resp': ..., ...}
#   pred       is a dict of signals, just like 'data' but containing 'pred'
#   mspec      is the modelspec data type, as was defined above
#   error      is a (scalar) measurement of the error between pred and resp

# Option 1: Use mean squared error when fitting:
metric = lambda data: nems.metrics.MSE(data['resp'], data['pred'])

# Option 2: Use log-likelihood, if you predicted a gaussian at each point
# metric = lambda data: nems.metrics.LogLikelihood(data['resp'], data['pred'], data['pred_stddev'])

# Option 3: Use some other metric that you think is better
# metric = lambda data: nems.metrics.coherence(data['resp'], data['pred'])

# Finally, define the evaluator and cost functions
# TODO: I think these can be boilerplate elsewhere
evaluator = lambda data, mspec : nems.model.Model(mspec).evaluate(data, mspec)
cost_fn = lambda mspec: metric(evaluator(est, mspec))

# ----------------------------------------------------------------------------
# FIT THE MODEL
# 
# GOAL: Sample from the parameter space to estimate the best parameter values 
# that accurately predict the data, or better still, the distributions of 
# parameter values that describe the data well. 

# Option 1: Fit using gradient descent (Fast)
fitter = nems.fitter.gradient_descent

# Option 2: Fit using simulated annealing (Slow, arguably gets stuck less often)
# fitter = nems.fitter.annealing

# Option 3: Fit using Metropolis-Hastings algorithm (Very slow, gives confidence bounds)
# fitter = nems.fitter.metropolis

# Option 4: Use an 10-fold cross-validated evaluator function (Slow, avoids overfitting)
# fitter = partial(nems.cross_validator.cross_validate_wrapper, gradient_descent, 10)

# Finally, run the fitter! (Takes a while)
modelspec_fitted = fitter(modelspec, cost_fn)


# ----------------------------------------------------------------------------
# PLOT, SAVE, AND PUBLISH RESULTS

# GOAL: Show how well your prediction described the actual, measured response.
# Show the confidence interval for each of the model parameters, if possible.

# Take your final measurements on the validation data set, and save!
results = {}
for m in nems.metrics.get_some_list_of_metric_functions:
    results[m.name] = m(val['resp'], evaluator(val, modelspec_fitted))

save_to_disk('/some/path/model1.json', json.dumps(modelspec_fitted))
save_to_disk('/some/path/model1_results.json', json.dumps(results))

# phi_distributions.plot('/some/other/path.png')

# Plot prediction and confidence intervals
# pred_EV = model.evaluate(phi_distributions.mean(), val)
# pred_10 = model.evaluate(phi_distributions.ppf(10), val)
# pred_90 = model.evaluate( phi_distributions.ppf(90), val)
# plot_signals('/some/path.png', pred_EV, pred_10, pred_90, ...)


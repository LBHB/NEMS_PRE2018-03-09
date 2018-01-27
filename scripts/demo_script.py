# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import json

from nems import initializers
from nems.analysis.api import fit_basic
from nems.recording import Recording

# ----------------------------------------------------------------------------
# DATA FETCHING

# GOAL: Get your data loaded into memory as a Recording object

# Method #1: Load the data from a local directory
rec = Recording.load('../signals/gus027b13_p_PPS/')
# TODO: temporary hack to avoid errors resulting from epochs not being defined.
for signal in rec.signals.values():
    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
# If there isn't a 'pred' signal yet, copy over 'stim' as the starting point.
# TODO: still getting a key error for 'pred' in fit_basic when
#       calling lambda on metric. Not sure why, since it's explicitly added.
rec.signals['pred'] = rec.signals['stim'].copy()


# Method #2: Load the data from baphy using the (incomplete, TODO) HTTP API:
# URL = "neuralprediction.org:3003/signals?batch=273&cellid=gus027b13-a1"
# rec = nems.utils.net.fetch_signals_over_http(URL)

# Method #3: Load the data from S3: (TODO)
# stimfile="https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"+cellid+"_NAT_stim_ozgf_c18_fs100.mat"
# respfile="https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"+cellid+"_NAT_resp_fs100.mat"
# rec = lbhb.fetch_signals_over_http(stimfile, respfile)

# Method #4: Load the data from a published jerb (TODO)

# Method #5: Create a Recording object from an array, manually (TODO)


# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.

# Method #1: Split based on time, where the first 80% is estimation data and
#            the last, last 20% is validation data.

# TODO: @Ivar -- per architecture.svg looked like this was going to be
#       handled inside an analysis by a segmentor? Designed fit_basic with
#       that in mind, so maybe this doesn't go here anymore, or I may have
#       had the wrong interpretation.    --jacob
#est, val = rec.split_at_time(0.8)

# Method #2: Split based on repetition number, rounded to the nearest rep.
# est, val = rec.split_at_rep(0.8)

# Method #3: Use the whole data set! (Usually for doing full dataset cross-val)
# est = rec
# val = rec


# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

# Method #1: create from "shorthand/default" keyword string
modelspec = initializers.from_keywords(rec, 'fir10x1_dexp1')
print('Modelspec was:')
print(modelspec)

results = [modelspec]

# Method #2: load a modelspec from disk
# modelspec = json.load('../modelspecs/wc2_fir10_dexp.json')

# Method #3: Load it from a published jerb (TODO)
# modelspec = ...

# Method #4: specify it manually (TODO)
# modelspec = ...


# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

# Option 1: Use gradient descent (Fast)
# TODO: @Ivar -- Raised question in fit_basic of whether fitter should be
#       exposed as argument to the analysis. Looks like that may have been
#       your original intention here? But I think if the fitter is exposed,
#       then the FitSpaceMapper also needs to be exposed since the type of
#       mapping needed may change depending on which fitter is use.
results = fit_basic(rec, modelspec)

# Option 2: Use simulated annealing (Slow, arguably gets stuck less often)
# results = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.annealing)

# Option 3: Use Metropolis algorithm (Very slow, gives confidence interval)
# results = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.metropolis)

# Option 4: Use 10-fold cross-validated evaluation
# fitter = partial(nems.cross_validator.cross_validate_wrapper, gradient_descent, 10)
# results = nems.analysis.fit_cv(est, modelspec, folds=10)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# GOAL: Save your results to disk. (BEFORE you screw it up trying to plot!)

# If only one result was returned, save it. But if multiple  modelspecs were
# returned, save all of them.

# TODO: ndarrays not json serializable, so need to decide the best way to
#       handle that.
#       Dealt with this before by always loading/saving with
#       son.dumps(array.tolist) and
#       some numpy method that interpreted ndarray from a string.
#       Would just need unpacker to deal with that I guess?
#       Otherwise script is working.  --jacob 1-26-18
if len(results) == 1:
    with open('../modelspecs/demo_script_model.json', mode='w+') as fp:
        json.dump(results[0], fp)
else:
    for i, m in enumerate(results):
        s = '../modelspecs/demo_script_model_{:04i}.json'.format(i)
        with open(s, mode='w+') as fp:
            json.dump(m, fp)


# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

# Optional: See how well your best result predicts the validation data set
# nems.plot.predictions(val, [results[0]])

# Optional: See how all the results predicted
# nems.plot.predictions(val, results)

# Optional: Compute the confidence intervals on your results
# nems.plot.confidence_intervals(val, results)

# Optional: View the prediction of the best result according to MSE
# nems.plot.best_estimator(val, results, metric=nems.metrics.mse)

# Optional: View the posterior parameter probability distributions
# nems.plot.posterior(val, results)


# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.

# TODO

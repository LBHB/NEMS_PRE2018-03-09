# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import json
import logging
import nems.modelspec as ms

from nems import initializers
from nems.analysis.api import fit_basic
from nems.recording import Recording

# ----------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.INFO)

signals_dir = '../signals'
modelspecs_dir = '../modelspecs'

# ----------------------------------------------------------------------------
# DATA FETCHING

# GOAL: Get your data loaded into memory as a Recording object

# Method #1: Load the data from a local directory
rec = Recording.load(os.path.join(signals_dir, 'gus027b13_p_PPS'))

# Method #2: Load the data from baphy using the (incomplete, TODO) HTTP API:
# URL = "http://neuralprediction.org:3003/by-batch/273/gus018c-a3"
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

est, val = rec.split_at_time(0.8)

# Method #2: Split based on repetition number, rounded to the nearest rep.
# est, val = rec.split_at_rep(0.8)

# Method #3: Use the whole data set! (Usually for doing full dataset cross-val)
# est = rec
# val = rec


# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

# Method #1: create from "shorthand" keyword string
modelspec = initializers.from_keywords('fir10x1_dexp1') 

# Method #2: load a modelspec from disk
# modelspec = ms.load_modelspec('../modelspecs/wc1_fir10x1_dexp1.json')

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
results = fit_basic(est, modelspec)

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

ms.save_modelspecs(modelspecs_dir, 'demo_script_model', results)

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

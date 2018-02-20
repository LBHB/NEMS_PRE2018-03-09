# A Template NEMS Script suitable for beginners
# Please see docs/architecture.svg for a visual diagram of this code

import os
import logging as log
import random
import numpy as np
import matplotlib.pyplot as plt
import nems
import nems.initializers
import nems.epoch as ep
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.analysis.api
import nems.utils
from nems.recording import Recording

# ----------------------------------------------------------------------------
# CONFIGURATION

log.basicConfig(level=log.INFO)

signals_dir = '../signals'
modelspecs_dir = '../modelspecs'

# ----------------------------------------------------------------------------
# DATA FETCHING

# GOAL: Get your data loaded into memory as a Recording object
log.info('Fetching data...')

# Method #1: Load the data from a local directory
rec = Recording.load(os.path.join(signals_dir, 'TAR010c-57-1'))

# Method #2: Load the data from baphy using the (incomplete, TODO) HTTP API:
# URL = "http://neuralprediction.org:3003/by-batch/273/gus018c-a3"
# rec = nems.utils.net.fetch_signals_over_http(URL)

# Method #3: Load the data from S3: (TODO)
# stimfile=("https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"
#           +cellid+"_NAT_stim_ozgf_c18_fs100.mat")
# respfile=("https://s3-us-west-2.amazonaws.com/nemspublic/sample_data/"
#           +cellid+"_NAT_resp_fs100.mat")
# rec = lbhb.fetch_signals_over_http(stimfile, respfile)

# Method #4: Load the data from a published jerb (TODO)

# Method #5: Create a Recording object from an array, manually (TODO)


# ----------------------------------------------------------------------------
# OPTIONAL PREPROCESSING
log.info('Preprocessing data...')

# Add a respavg signal to the recording
rec = preproc.add_average_sig(rec, signal_to_average='resp',
                              new_signalname='respavg',
                              epoch_regex='^STIM_')

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.

log.info('Withholding validation set data...')

# Method #0: Try to guess which stimuli have the most reps, use those for val
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

# Method #1: Split based on time, where the first 80% is estimation data and
#            the last, last 20% is validation data.
# est, val = rec.split_at_time(0.8)

# Method #2: Split based on repetition number, rounded to the nearest rep.
# est, val = rec.split_at_rep(0.8)

# Method #3: Use the whole data set! (Usually for doing n-fold cross-val)
# est = rec
# val = rec


# ----------------------------------------------------------------------------
# INITIALIZE MODELSPEC

# GOAL: Define the model that you wish to test

log.info('Loading modelspec(s)...')

# Method #1: create from "shorthand" keyword string
# modelspec = nems.initializers.from_keywords('wc18x1_lvl1_fir10x1')

# Method #2: Load modelspec(s) from disk
# TODO: allow selection of a specific modelspec instead of ALL models for this data!!!!
modelspecs = ms.load_modelspecs(modelspecs_dir, 'TAR010c-57-1')

# Method #3: Load it from a published jerb (TODO)
# modelspec = ...

# Method #4: specify it manually (TODO)
# modelspec = ...

# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

# Option 1: Use gradient descent on whole data set(Fast)
#results = nems.analysis.api.fit_basic(est, modelspec)

# Option 2: Split the est data into 10 pieces, fit them, and average
# results = nems.analysis.api.fit_random_subsets(est, modelspec, nsplits=10)
# result = average(results...)

# Option 3: Fit 10 jackknifes of the data, and return all of them.
# results = nems.analysis.api.fit_jackknifes(est, modelspec, njacks=10)

# Option 4: Divide estimation data into 10 subsets; fit all sets separately
# results = nems.analysis.api.fit_subsets(est, modelspec, nsplits=3)

# Option 5: Start from random starting points 10 times
# results = nems.analysis.api.fit_from_priors(est, modelspec, ntimes=10)

# TODO: Perturb around the modelspec to get confidence intervals

# TODO: Use simulated annealing (Slow, arguably gets stuck less often)
# results = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.annealing)

# TODO: Use Metropolis algorithm (Very slow, gives confidence interval)
# results = nems.analysis.fit_basic(est, modelspec,
#                                   fitter=nems.fitter.metropolis)

# TODO: Use 10-fold cross-validated evaluation
# fitter = partial(nems.cross_validator.cross_validate_wrapper, gradient_descent, 10)
# results = nems.analysis.fit_cv(est, modelspec, folds=10)


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# GOAL: Save your results to disk. (BEFORE you screw it up trying to plot!)

# log.info('Saving Results')

# ms.save_modelspecs(modelspecs_dir, results)

# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

log.info('Generating summary plot...')

# Generate a summary plot
nplt.plot_summary(val, modelspecs)

# Optional: See how well your best result predicts the validation data set
# nems.plot.predictions(val, [results[0]])

# Optional: See how all the results predicted
# nems.plot.predictions(val, results)

# Optional: Compute the confidence intervals on your results
# nems.plot.confidence_intervals(val, results) # TODO

# Optional: View the prediction of the best result according to MSE
# nems.plot.best_estimator(val, results, metric=nems.metrics.mse) # TODO

# Optional: View the posterior parameter probability distributions
# nems.plot.posterior(val, results) # TODO


#nplt.pred_vs_act_scatter(val, one_modelspec, ms.evaluate, ax=ax1)
#nplt.pred_vs_act_psth(val, one_modelspec, ms.evaluate, ax=ax2)
#nplt.pred_vs_act_psth_smooth(val, one_modelspec, ms.evaluate, ax=ax3)

# Pause before quitting
plt.show()

# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.b

# TODO

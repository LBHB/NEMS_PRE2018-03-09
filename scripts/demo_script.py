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
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize

# ----------------------------------------------------------------------------
# CONFIGURATION

log.basicConfig(level=log.INFO)

signals_dir = '../signals'
modelspecs_dir = '../modelspecs'

# ----------------------------------------------------------------------------
# DATA LOADING

# GOAL: Get your data loaded into memory as a Recording object
log.info('Loading data...')

# Method #1: Load the data from a local directory
rec = Recording.load(os.path.join(signals_dir, 'TAR010c-18-1'))

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

# Add a respavg signal to the recording now, so we don't have to do it later
# on both the est and val sets seperately.
rec = preproc.add_average_sig(rec, signal_to_average='resp',
                              new_signalname='resp', # NOTE: ADDING AS RESP NOT RESPAVG FOR TESTING
                              epoch_regex='^STIM_')

# ----------------------------------------------------------------------------
# DATA WITHHOLDING

# GOAL: Split your data into estimation and validation sets so that you can
#       know when your model exhibits overfitting.

log.info('Withholding validation set data...')

# Method #0: Try to guess which stimuli have the most reps, use those for val
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

# Optional: Take nanmean of ALL occurrences of all signals
# est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
# val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

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

log.info('Initializing modelspec(s)...')

# Method #1: create from "shorthand" keyword string
modelspec = nems.initializers.from_keywords('wc18x1_lvl1_fir15x1_dexp1')

# Method #2: Load modelspec(s) from disk
# TODO: allow selection of a specific modelspec instead of ALL models for this data!!!!
# results = ms.load_modelspecs(modelspecs_dir, 'TAR010c-57-1')

# Method #3: Load it from a published jerb (TODO)
# results = ...

# ----------------------------------------------------------------------------
# RUN AN ANALYSIS

# GOAL: Fit your model to your data, producing the improved modelspecs.
#       Note that: nems.analysis.* will return a list of modelspecs, sorted
#       in descending order of how they performed on the fitter's metric.

log.info('Fitting Modelspec(s)...')

# Option 1: Use gradient descent on whole data set(Fast)
# modelspecs = nems.analysis.api.fit_basic(est, modelspec)

# Fit on whole recording! Not just est and val.
modelspecs = nems.analysis.api.fit_basic(est, modelspec, fitter=scipy_minimize,
                                         metric=lambda data: nems.metrics.api.nmse(
                                             {'pred': data.get_signal('pred').as_continuous(),
                                              'resp': data.get_signal('resp').as_continuous()}
                                         ),)

# Option 2: Split the est data into 10 pieces, fit them, and average
# modelspecs = nems.analysis.api.fit_random_subsets(est, modelspec, nsplits=10)
# result = average(modelspecs...)

# Option 3: Fit 4 jackknifes of the data, and return all of them.
# modelspecs = nems.analysis.api.fit_jackknifes(est, modelspec, njacks=4)

# Option 4: Divide estimation data into 10 subsets; fit all sets separately
# modelspecs = nems.analysis.api.fit_subsets(est, modelspec, nsplits=3)

# Option 5: Start from random starting points 10 times
# modelspecs = nems.analysis.api.fit_from_priors(est, modelspec, ntimes=10)

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


# ----------------------------------------------------------------------------
# SAVE YOUR RESULTS

# GOAL: Save your results to disk. (BEFORE you screw it up trying to plot!)

log.info('Saving Results...')

ms.save_modelspecs(modelspecs_dir, modelspecs)

# ----------------------------------------------------------------------------
# GENERATE SUMMARY STATISTICS

log.info('Generating summary statistics...')





# ----------------------------------------------------------------------------
# GENERATE PLOTS

# GOAL: Plot the predictions made by your results vs the real response.
#       Compare performance of results with other metrics.

log.info('Generating summary plot...')

# Generate a summary plot
nplt.plot_summary(val, modelspecs)

# Optional: See how well your best result predicts the validation data set
# nems.plot.predictions(val, [results[0]]) # TODO

# Optional: See how all the results predicted
# nems.plot.predictions(val, results) # TODO

# Optional: Compute the confidence intervals on your results
# nems.plot.confidence_intervals(val, results) # TODO

# Optional: View the prediction of the best result according to MSE
# nems.plot.best_estimator(val, results, metric=nems.metrics.mse) # TODO

# Optional: View the posterior parameter probability distributions
# nems.plot.posterior(val, results) # TODO

# nplt.pred_vs_act_scatter(val, one_modelspec, ms.evaluate, ax=ax1)
#  nplt.pred_vs_act_psth(val, one_modelspec, ms.evaluate, ax=ax2)
#nplt.pred_vs_act_psth_smooth(val, one_modelspec, ms.evaluate, ax=ax3)

# Pause before quitting
plt.show()

# ----------------------------------------------------------------------------
# SHARE YOUR RESULTS

# GOAL: Upload your resulting models so that you can see how well your model
#       did relative to other peoples' models. Save your results to a DB.b

# TODO

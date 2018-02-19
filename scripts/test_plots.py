import os
import random

import matplotlib.pyplot as plt

import nems.modelspec as ms
import nems.plots.api as nplt
from nems.recording import Recording

# specify directories for loading data and fitted modelspec
signals_dir = '../signals'
modelspecs_dir = '../modelspecs'

# load the data, split to est and val, load modelspecs
# TODO: How to ensure that est, val split are the same as they
#       were for when the modelspec was fitted?
#       Will this information be in the modelspec metadata?
#       Sometihng like meta: {'segmentor': ('split_at_time', 0.8)}?
rec = Recording.load(os.path.join(signals_dir, 'TAR010c-57-1'))
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
# est, val = rec.split_at_time(0.8)
loaded_modelspecs = ms.load_modelspecs(modelspecs_dir, 'TAR010c-57-1')
stim = val['stim']
resp = val['resp']
pred = ms.evaluate(val, loaded_modelspecs[0])['pred']

fig = plt.figure(figsize=(12,12))
plt.subplot(111)
nplt.plot_scatter(resp, pred, title=rec.name)
fig.show()
plt.show()
exit()

# add some fake epochs for testing since split is messing them up
#stim.epochs = stim.trial_epochs_from_occurrences(occurrences=377)
#resp.epochs = resp.trial_epochs_from_occurrences(occurrences=377)


# TODO: these need work. hard to interpret with all trials etc present
#       at once.
# use defaults for all plot functions using the 'high-level' plotting functions
#plot_fns = [nplt.pred_vs_act_scatter, nplt.pred_vs_act_psth]
#frozen_fns = nplt.freeze_defaults(plot_fns, val, loaded_modelspecs[0],
#                                  ms.evaluate)
#fig = nplt.simple_grid(frozen_fns, nrows=len(plot_fns))
#print("Signals with all epochs included")
#fig.show()



# plot prediction versus response for three randomly selected occurrences
# of 'TRIAL' epoch, using 'low-level' plotting functions.
evaluated_spec = ms.evaluate(val, loaded_modelspecs[0])
pred = evaluated_spec['pred']
total_o = pred.count_epoch('TRIAL')
# TODO: trim pre and post stim silence? ~1/3 of spectrogram is empty
for i in range(3):
    fig = plt.figure(figsize=(12,6))
    o = random.randrange(total_o)
    plt.subplot(211)
    nplt.spectrogram_from_epoch(stim, 'TRIAL', occurrence=o)
    plt.title("Trial {}".format(o))
    plt.subplot(212)
    nplt.timeseries_from_epoch([pred, resp], 'TRIAL', occurrence=o)
    plt.ylabel('Firing Rate')
    plt.tight_layout()
    fig.show()

fig = plt.figure(figsize=(12, 3))
plt.subplot(111)
nplt.weight_channels_heatmap(loaded_modelspecs[0])
fig.show()

# Compare predictions from a few different modelspecs against each other
# for 3 random occurrences
fig = plt.figure(figsize=(12, 9))
for i in range(3):
    plt.subplot(3, 1, i+1)
    preds = nplt.get_predictions(val, loaded_modelspecs, ms.evaluate)
    signals = [resp]
    signals.extend(preds)
    o = random.randrange(total_o)
    nplt.timeseries_from_epoch(signals, 'TRIAL', occurrence=o,
                               ylabel='Firing Rate')
plt.tight_layout()
fig.show()

# Compare weight channels coefficients for a few different modelspecs
nplt.weight_channels_heatmaps(loaded_modelspecs, figsize=(12,9))

plt.show()

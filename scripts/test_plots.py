import os
import random
from functools import partial
import matplotlib.pyplot as plt
import nems.modelspec as ms
import nems.plots.api as nplt
from nems.recording import Recording

# specify directories for loading data and fitted modelspec
signals_dir = '../signals'
modelspecs_dir = '../modelspecs'

# load the data, split to est and val, load modelspecs
rec = Recording.load(os.path.join(signals_dir, 'TAR010c-57-1'))
est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
# est, val = rec.split_at_time(0.8)

# Load some modelspecs and create their predictions
modelspecs = ms.load_modelspecs(modelspecs_dir, 'TAR010c-57-1')
pred = [ms.evaluate(val, m)['pred'] for m in modelspecs]

# Shorthands for unchanging signals
stim = val['stim']
resp = val['resp']

def plot_layout(plot_fn_struct):
    # Count how many plot functions we want
    nrows = len(plot_fn_struct)
    ncols = max([len(row) for row in plot_fn_struct])
    # Set up the subplots
    fig = plt.figure(figsize=(12,12))
    for r, row in enumerate(plot_fn_struct):
        for c, plotfn in enumerate(row):
            colspan = max(1, int(ncols / len(row)))
            ax = plt.subplot2grid((nrows, ncols), (r, c), colspan=colspan)  
            plotfn(ax=ax)
    return fig

# Test the layout
def my_scatter(ax): nplt.plot_scatter(resp, pred[0], ax=ax, title=rec.name)


#
sigs = [resp]
sigs.extend(pred)
fig = plot_layout([[my_scatter, my_scatter],
                   [lambda ax : nplt.spectrogram_from_epoch(stim, 'TRIAL', ax=ax, occurrence=2)],
                   [lambda ax : nplt.timeseries_from_epoch(sigs, 'TRIAL', ax=ax, occurrence=2)]])

fig.tight_layout()
fig.show()

#     nplt.timeseries_from_epoch(signals, 'TRIAL', occurrence=o,
#                                ylabel='Firing Rate')

# # Plot three predictions against the real response. 
# # Compare predictions from a few different modelspecs against each other
# # for 3 random occurrences
# fig = plt.figure(figsize=(12, 9))
# for i in range(3):
#     nplt.timeseries_from_epoch(signals, 'TRIAL', occurrence=o,
#                                ylabel='Firing Rate')




plt.show()
exit()

################################################################################
# TODO: How to ensure that est, val split are the same as they
#       were for when the modelspec was fitted?
#       Will this information be in the modelspec metadata?
#       Sometihng like meta: {'segmentor': ('split_at_time', 0.8)}?

# TODO: Fix problem with splitting messing up epochs. This was workaround:
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

 
# Compare weight channels coefficients for a few different modelspecs
nplt.weight_channels_heatmaps(loaded_modelspecs, figsize=(12,9))

plt.show()

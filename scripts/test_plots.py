import os

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
rec = Recording.load(os.path.join(signals_dir, 'gus027b13_p_PPS'))
est, val = rec.split_at_time(0.8)
loaded_modelspecs = ms.load_modelspecs(modelspecs_dir, 'demo_script_model')
# add some fake epochs for testing
stim = val['stim']
resp = val['resp']
stim.epochs = stim.trial_epochs_from_occurrences(occurrences=47)
resp.epochs = resp.trial_epochs_from_occurrences(occurrences=47)

# use defaults for all plot functions using the 'high-level' plotting functions
# TODO: these definitely still need tweaking.
plot_fns = [nplt.pred_vs_act_scatter, nplt.pred_vs_act_psth]
frozen_fns = nplt.freeze_defaults(plot_fns, val, loaded_modelspecs[0],
                                  ms.evaluate)
fig = nplt.simple_grid(frozen_fns, nrows=len(plot_fns))
print("Signals with all epochs included")
fig.show()

# plot a specific epoch and channel using the 'low-level' functions
# and manual subplotting.
# TODO: splitting up into epochs is definitely much more readable.
#       maybe need to just make something like this the default behavior
#       for a 'quick_plot' type function?
fig = plt.figure(figsize=(12, 9))
evaluated_spec = ms.evaluate(val, loaded_modelspecs[0])
pred = evaluated_spec['pred']
for i in range(7):
    plt.subplot(7, 1, i+1)
    j = i*6
    nplt.timeseries_from_epoch([pred, resp], 'trial', occurrence=j)
plt.ylabel('Firing Rate')
plt.tight_layout()
fig.show()

import os

import nems.modelspec as ms
import nems.plots.api as nplt
from nems.recording import Recording

signals_dir = '../signals'
modelspecs_dir = '../modelspecs'

rec = Recording.load(os.path.join(signals_dir, 'gus027b13_p_PPS'))
est, val = rec.split_at_time(0.8)
loaded_modelspecs = ms.load_modelspecs(modelspecs_dir, 'demo_script_model')

plot_fns = [nplt.pred_vs_act_scatter, nplt.pred_vs_act_psth]
frozen_fns = nplt.freeze_defaults(plot_fns, val, loaded_modelspecs[0],
                                  ms.evaluate)
fig = nplt.simple_grid(frozen_fns, nrows=len(plot_fns))
fig.show()
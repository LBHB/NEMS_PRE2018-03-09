from .assemble import simple_grid, freeze_defaults
from .stim import plot_stim_occurrence
from .spectrogram import (plot_spectrogram, spectrogram_from_signal,
                          spectrogram_from_epoch)
from .timeseries import timeseries_from_signals, timeseries_from_epoch
from .heatmap import weight_channels_heatmap
from .pred_vs_actual import (pred_vs_act_scatter, pred_vs_act_psth,
                             pred_vs_act_psth_smooth)
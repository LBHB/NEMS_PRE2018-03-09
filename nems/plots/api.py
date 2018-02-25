from .assemble import (simple_grid, freeze_defaults, get_predictions,
                       plot_layout, combine_signal_channels, pad_to_signals)
from .stim import plot_stim_occurrence
from .scatter import plot_scatter, plot_scatter_smoothed
from .summary import plot_summary
from .spectrogram import (plot_spectrogram, spectrogram_from_signal,
                          spectrogram_from_epoch)
from .timeseries import timeseries_from_signals, timeseries_from_epoch
from .heatmap import weight_channels_heatmap, fir_heatmap, strf_heatmap
from .pred_vs_actual import (pred_vs_act_scatter, pred_vs_act_psth,
                             pred_vs_act_psth_smooth)
from .file import save_figure, load_figure_img, load_figure_bytes
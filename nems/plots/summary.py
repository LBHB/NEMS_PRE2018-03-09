from functools import partial
from nems.plots.assemble import plot_layout
from nems.plots.heatmap import weight_channels_heatmap
from nems.plots.scatter import plot_scatter
from nems.plots.spectrogram import spectrogram_from_epoch
from nems.plots.timeseries import timeseries_from_epoch
import nems.modelspec as ms


def plot_summary(rec, modelspecs):
    '''
    Plots a summary of the modelspecs and their performance predicting on rec.
    '''
    stim = rec['stim']
    resp = rec['resp']
    respavg = rec['respavg']

    # Make predictions on the data set using the modelspecs
    pred = [ms.evaluate(rec, m)['pred'] for m in modelspecs]

    # Example of how to plot a complicated thing:
    occurrence = 0
    def my_scatter(idx, ax): plot_scatter(resp, pred[idx], ax=ax, title=rec.name)
    def my_spectro(ax): spectrogram_from_epoch(stim, 'TRIAL', ax=ax, occurrence=occurrence)
    sigs = [respavg]
    sigs.extend(pred)
    def my_timeseries(ax) : timeseries_from_epoch(sigs, 'TRIAL', ax=ax, occurrence=occurrence)
    def my_strf(idx, ax) : weight_channels_heatmap(modelspecs[idx], ax=ax)
    
    fig = plot_layout([[partial(my_scatter, 0), partial(my_scatter, 1), partial(my_scatter, 2)],
                       [partial(my_strf, 0), partial(my_strf, 1), partial(my_strf, 2)],
                       [my_spectro],                    
                       [my_timeseries]])

    fig.tight_layout()
    fig.show()
    

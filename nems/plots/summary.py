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
    resp = rec['respavg'] if 'respavg' in rec.signals else rec['resp']

    # Make predictions on the data set using the modelspecs
    pred = [ms.evaluate(rec, m)['pred'] for m in modelspecs]

    sigs = [resp]
    sigs.extend(pred)

    # Example of how to plot a complicated thing:
    occurrence = 0

    def my_scatter(idx, ax): plot_scatter(resp, pred[idx], ax=ax, title=rec.name)
    def my_spectro(ax): spectrogram_from_epoch(stim, 'TRIAL', ax=ax, occurrence=occurrence)
    def my_timeseries(ax) : timeseries_from_epoch(sigs, 'TRIAL', ax=ax, occurrence=occurrence)
    def my_strf(idx, ax) : weight_channels_heatmap(modelspecs[idx], ax=ax)
    
    def make_partials(fn, items):
        partials = [partial(fn, i) for i in range(len(items))]
        return partials

    if len(modelspecs) <= 10:
        fig = plot_layout([make_partials(my_scatter, modelspecs),
                           make_partials(my_strf, modelspecs),
                           [my_spectro],                    
                           [my_timeseries]])
    else:
        # Don't plot the scatters/strfs when you have more than 10
        fig = plot_layout([[my_spectro],                    
                           [my_timeseries]])

    fig.tight_layout()
    fig.show()
    

import numpy as np
import matplotlib.pyplot as plt

from nems.signal import Signal

def plot_timeseries(times, values, xlabel='Time', ylabel='Value', legend=None, ax=None):
    '''
    Plots a simple timeseries with one line for each pair of
    time and value vectors.
    Lines will be auto-colored according to matplotlib defaults.

    times : list of vectors
    values : list of vectors
    xlabel : str
    ylabel : str
    legend : list of strings
    TODO: expand this doc  -jacob 2-17-18
    '''
    for t, v in zip(times, values):
        ax.plot(t, v)

    ax.margins(x=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(legend)


def timeseries_from_signals(signals, channel=0, xlabel='Time', ylabel='Value',
                            ax=None, concat=False, i=None, j=None):
    
    # TODO: extract this to separate utility function and rename channels
    #       accordingly.
    # NOTE: should channel renaming be done in the signals method anyway?
    
    # Starting with index i, signals through (but not including) j will
    # be concatenated channel-wise to signals[i].
    if concat:
        signals[i:j] = Signal.concatenate_channels(signals[i:j])
    
    legend = [s.name for s in signals]
    times = []
    values = []
    for s in signals:
        # Get values from specified channel
        value_vector = s.as_continuous()[channel]
        # Convert indices to absolute time based on sampling frequency
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
    plot_timeseries(times, values, xlabel, ylabel, legend, ax=ax)

def timeseries_from_epoch(signals, epoch, occurrence=0, channel=None,
                          xlabel='Time', ylabel='Value', ax=None,
                          concat=False, i=None, j=None):
    # Starting with index i, signals through (but not including) j will
    # be concatenated channel-wise to signals[i].
    if concat:
        signals[i:j] = Signal.concatenate_channels(signals[i:j])
        
    legend = [s.name for s in signals]
    times = []
    values = []
    for s in signals:
        # Get occurrences x chans x time
        extracted = s.extract_epoch(epoch)
        # Get values from specified occurrence and channel
        if channel is None:
            # all channels
            value_vector = extracted[occurrence].T
        else:
            value_vector = extracted[occurrence][channel]
        # Convert bins to time (relative to start of epoch)
        # TODO: want this to be absolute time relative to start of data?
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
    plot_timeseries(times, values, xlabel, ylabel, legend, ax=ax)

import numpy as np
import matplotlib.pyplot as plt

def plot_timeseries(times, values, ax=None, xlabel='Time', ylabel='Value',
                    legend=None):
    """Plots a simple timeseries with one line for each pair of
    time and value vectors.
    Lines will be auto-colored according to matplotlib defaults.

    times : list
    values : list
    ax : matplotlib axes object
    xlabel : str
    ylabel : str
    legend : list
    TODO: expand this doc  -jacob 2-17-18
    """
    if ax:
        plt.sca(ax)

    for t, v in zip(times, values):
        plt.plot(t, v)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend(legend)

def timeseries_from_signals(signals, channel=0, ax=None, xlabel='Time',
                            ylabel='Value'):
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
    plot_timeseries(times, values, ax, xlabel, ylabel, legend)

def timeseries_from_epoch(signals, epoch, occurrence=0, channel=0,
                          ax=None, xlabel='Time', ylabel='Value'):
    legend = [s.name for s in signals]
    times = []
    values = []
    for s in signals:
        # Get occurrences x chans x time
        extracted = s.extract_epoch(epoch)
        # Get values from specified occurrence and channel
        value_vector = extracted[occurrence][channel]
        # Convert bins to time (relative to start of epoch)
        # TODO: want this to be absolute time relative to start of data?
        time_vector = np.arange(0, len(value_vector)) / s.fs
        times.append(time_vector)
        values.append(value_vector)
    plot_timeseries(times, values, ax, xlabel, ylabel, legend)

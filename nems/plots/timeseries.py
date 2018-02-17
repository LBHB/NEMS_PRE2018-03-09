import numpy as np
import matplotlib.pyplot as plt

def plot_timeseries(signals, ax=None, xlabel='Time', ylabel='Signal Value'):
    """Plots a simple timeseries with one line for each signal in signals.
    Data is based on the 2d array returned by signal.as_continuous().

    Lines will be auto-colored according to matplotlib defaults.
    If a matplotlib axes object is provided by the ax keyword argument,
    pyplot will use that as the current axes for plotting.
    """
    if ax:
        plt.sca(ax)

    for s in signals:
        # Convert indices to absolute time based on sampling frequency
        chans, bins = s.shape
        indices = range(bins)
        times = [i/s.fs for i in indices]
        x = np.array(times)
        # Get values from first channel
        # TODO: what to do about multiple channels?
        #       Shouldn't matter for pred_vs_act usage since only 1 channel,
        #       But for the general case need... what? multiple plots? 3d plot?
        values = s.as_continuous()[0, :]
        y = np.array(np.squeeze(values))
        plt.plot(x, y)

    plt.legend([s.name for s in signals])
    plt.xlabel(xlabel)
    # TODO: Best way to get more information for this label?
    plt.ylabel(ylabel)
import numpy as np
import matplotlib.pyplot as plt


def plot_scatter(sig1, sig2, ax=None, title=None, xlabel=None, ylabel=None, legend=True):
    '''
    Uses the channels of sig1 to place points along the x axis, and channels of
    sig2 for distances along the y axis. If sig1 has one channel but sig2 has
    multiple channels, then all of sig2's channels will be plotted against the
    values from sig1. If sig1 has more than 1 channel, then sig2 must have the
    same number of channels, because then XY coordinates will be determined from
    both sig1 and sig2.

    Optional arguments:
        ax
        xlabel
        ylabel
        legend   
    '''
    if sig1.nchans > 1 and sig1.nchans != sig2.nchans:
        m = 'sig1 and sig2 must have same number of chans if sig1 is multichannel'
        raise ValueError(m)

    if ax:
        plt.sca(ax)

    m1 = sig1.as_continuous()
    m2 = sig2.as_continuous()
    for i in range(sig2.nchans):
        if sig1.nchans > 1:
            x = np.array(np.squeeze(m1[i, :]))
        else:
            x = np.array(np.squeeze(m1[0, :]))
        y = np.array(np.squeeze(m2[i, :]))
        chan_name = 'Channel {}'.format(i) if not sig2.chans else sig2.chans[i]
        plt.scatter(x, y, label=chan_name)

    xlabel = xlabel if xlabel else sig1.name
    plt.xlabel(xlabel)

    ylabel = ylabel if ylabel else sig2.name
    plt.ylabel(ylabel)

    if legend and sig2.nchans > 1
        plt.legend(loc='best')

    if title:
        plt.title(title)

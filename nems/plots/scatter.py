import numpy as np
import matplotlib.pyplot as plt

def plot_scatter(signals, ax=None, xlabel='Signal One', ylabel='Signal Two'):
    if ax:
        plt.sca(ax)

    # TODO: What to do if there are more than two signals?
    #       Currently hardcoded to only use first two
    # TODO: What to do if more than one channel?
    x_values = signals[0].as_continuous()[0, :]
    x = np.array(np.squeeze(x_values))
    y_values = signals[1].as_continuous()[0, :]
    y = np.array(np.squeeze(y_values))

    plt.plot(x, y, 'ko')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
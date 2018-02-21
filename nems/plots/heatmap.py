import matplotlib.pyplot as plt
import numpy as np


def plot_heatmap(array, xlabel='Dim One', ylabel='Dim Two',
                 ax=None, cmap=None, clim=None):
    # Make sure array is converted to ndarray if passed as list
    array = np.array(array)

    if not clim:
        mmax = np.max(np.abs(array.reshape(-1)))
        clim = [-mmax, mmax]

    ax.imshow(array, aspect='auto', origin='lower',
              cmap=plt.get_cmap('jet'),
              clim=clim,
              interpolation='none')

    # Force integer tick labels
    y, x = array.shape
    ax.set_yticks(np.arange(y))
    ax.set_xticks(np.arange(x))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # Set the color bar
    # cbar = ax.colorbar()
    # cbar.set_label('Gain')


def _get_wc_coefficients(modelspec):
    for m in modelspec:
        if 'weight_channels' in m['fn']:
            return m['phi']['coefficients']


def _get_fir_coefficients(modelspec):
    for m in modelspec:
        if 'fir_filter' in m['fn']:
            return m['phi']['coefficients']


def weight_channels_heatmap(modelspec, ax=None):
    coefficients = _get_wc_coefficients(modelspec)
    plot_heatmap(coefficients, xlabel='Channel In', ylabel='Channel Out', ax=ax)


def fir_heatmap(modelspec, ax=None):
    coefficients = _get_fir_coefficients(modelspec)
    plot_heatmap(coefficients, xlabel='Time Bin', ylabel='Channel In', ax=ax)

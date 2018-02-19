import matplotlib.pyplot as plt
import numpy as np

# TODO: when plotting multiple heatmaps, sync up the color scales
def plot_heatmap(array, ax=None, xlabel='Dim One', ylabel='Dim Two'):
    # Make sure array is converted to ndarray if passed as list
    array = np.array(array)
    # TODO: copied from nems master but not 100% sure what this does.
    #       obviously grabbing a max but which axis? is it flattening it first?
    mmax = np.max(np.abs(array.reshape(-1)))
    plt.imshow(array, aspect='auto', origin='lower', cmap=plt.get_cmap('jet'),
               clim=[-mmax,mmax], interpolation='none')
    # Force integer tick labels
    y, x = array.shape
    plt.yticks(np.arange(y))
    plt.xticks(np.arange(x))
    cbar = plt.colorbar()
    cbar.set_label('Gain')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

def weight_channels_heatmap(modelspec):
    coefficients = _get_wc_coefficients(modelspec)
    plot_heatmap(coefficients, xlabel='Channel In', ylabel='Channel Out')

def _get_wc_coefficients(modelspec):
    for m in modelspec:
        if 'weight_channels' in m['fn']:
            return m['phi']['coefficients']

def fir_heatmap(modelspec):
    raise NotImplementedError

def _get_fir_coefficients(modelspec):
    raise NotImplementedError
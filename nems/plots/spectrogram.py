import scipy.signal as sps
import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_spectrogram(array, fs=None):
    plt.imshow(array, aspect='auto', origin='lower', interpolation='none')
    
    # TODO: Couldn't get this to work, but would be nice for consistency.
    #       Just want label for time bin 100 to be 100/fs, but the labels have
    #       some mpl-specific 'text' type that's a pain to work with.
    #       Probably need to just make custom tick locations as well but
    #       didn't want to sink too much time into it at the moment.
    #       -jacob 2-19-18
    # Allow x-axis tick label override to display as real time instead of
    # time bin indices.
    #if fs is not None:
    #    locs, labels = plt.xticks()
    #    labels = np.array(labels, dtype=np.int)
    #    plt.xticks(locs, labels/fs)

    cbar = plt.colorbar()
    cbar.set_label('Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Channel')

def spectrogram_from_signal(signal):
    array = signal.as_continuous()
    plot_spectrogram(array, fs=signal.fs)

def spectrogram_from_epoch(signal, epoch, occurrence=0):
    extracted = signal.extract_epoch(epoch)
    array = extracted[occurrence]
    plot_spectrogram(array, fs=signal.fs)
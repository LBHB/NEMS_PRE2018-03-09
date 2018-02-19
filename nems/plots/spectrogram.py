import scipy.signal as sps
import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_spectrogram(array):
    plt.imshow(array, aspect='auto', origin='lower', interpolation='none')
    cbar = plt.colorbar()
    cbar.set_label('Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Channel')

def spectrogram_from_signal(signal):
    array = signal.as_continuous()
    plot_spectrogram(array)

def spectrogram_from_epoch(signal, epoch, occurrence=0):
    extracted = signal.extract_epoch(epoch)
    array = extracted[occurrence]
    plot_spectrogram(array)
import scipy.signal as sps
import scipy as scp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Plot functions need a recording (data to work with),
#              a fitted modelspec (what parameters to use),
#       and an evaluator function (how to transform the data before plotting).

# TODO: Eventually want to extract out figsize as a config setting?
FIGSIZE = (12, 4)

def plot_spectrogram(recording, modelspec, evaluator, sub_idx=None):
    raise NotImplementedError

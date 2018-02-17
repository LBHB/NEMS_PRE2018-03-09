import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (12, 4)

def plot_stim_occurrence(recording, modelspec, evaluator, transform_idx=-1,
                         occurrence_idx=0, epoch_name='stim',
                         signal_name='stim'):
    """
    TODO: doc
    """
    # TODO: needs testing
    signal = recording[signal_name]
    data = signal.extract_epoch(epoch_name)
    squeezed = np.squeeze(data[occurrence_idx, :, :])
    flipped = squeezed.transpose()
    plt.plot(flipped)

    #last_transform_fn = modelspec[transform_idx]['fn']
    #plt.title('')
    plt.xlabel('Time (ms)')
    plt.ylabel('Channel?')
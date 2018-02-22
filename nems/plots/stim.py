import numpy as np
import matplotlib.pyplot as plt

def plot_stim_occurrence(recording, modelspec, evaluator, transform_idx=-1,
                         occurrence_idx=0, epoch_name='stim',
                         signal_name='stim'):
    """
    TODO: doc
    """
    raise NotImplementedError
    # TODO: needs refactoring after redoing other plots.
    #       Also, does this one even get used? Probably check that before
    #       bothering with it again.   -jacob 2-19-18
    signal = recording[signal_name]
    data = signal.extract_epoch(epoch_name)
    squeezed = np.squeeze(data[occurrence_idx, :, :])
    flipped = squeezed.transpose()
    plt.plot(flipped)

    #last_transform_fn = modelspec[transform_idx]['fn']
    #plt.title('')
    plt.xlabel('Time (ms)')
    plt.ylabel('Channel?')
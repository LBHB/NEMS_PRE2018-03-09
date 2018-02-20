import numpy as np
import nems.epoch as ep


def add_average_sig(rec, signal_to_average='resp', new_signalname='respavg', epoch_regex='^STIM_'):
    '''
    Returns a recording with a new signal: the response average.
    TODO: Docs
    '''
    
    # 1. Fold matrix over all stimuli, returning a dictionary where keys are stimuli 
    #    and each value in the dictionary is (reps X cell X bins)
    epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
    folded_matrix = rec[signal_to_average].extract_epochs(epochs_to_extract)
    
    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrix.keys():
        per_stim_psth[k] = np.nanmean(folded_matrix[k], axis=0)
        
    # 3. Invert the folding to unwrap the psth back out into a predicted spike_dict by 
    #   replacing all epochs in the signal with their average (psth)
    respavg = rec[signal_to_average].replace_epochs(per_stim_psth)
    respavg.name = new_signalname

    # 4. Now add the signal to the recording
    newrec = rec.copy()
    newrec.add_signal(respavg)

    return newrec


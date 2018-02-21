import numpy as np
import nems.epoch as ep
import pandas as pd
import nems.signal as signal

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


def convert_to_average_sig(rec, epoch_regex='^STIM_'):
    '''
    Returns a recording with a new signal: the response average.
    TODO: Docs
    '''
        
    # Create new recording
    newrec = rec.copy()
    
    # iterate through each signal
    for signal_name,signal_to_average in rec.signals.items():
        # 1. Find matching epochs
        epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)
        
        # 2. Fold matrix over all stimuli, returning a dictionary where keys are stimuli 
        #    and each value in the dictionary is (reps X cell X bins)
        folded_matrix = signal_to_average.extract_epochs(epochs_to_extract)
        
        # 3. Average over all reps of each stim and append to data matrix
        fs=signal_to_average.fs
        epochs = None
        data = np.zeros([signal_to_average.nchans,0])
        current_time=0
        
        for k in folded_matrix.keys():
            per_stim_psth = np.nanmean(folded_matrix[k], axis=0)
            data=np.concatenate((data,per_stim_psth),axis=1)
            
            epoch=current_time+np.array([[0,per_stim_psth.shape[1]/fs]])
            
            df = pd.DataFrame(np.tile(epoch,[2,1]), columns=['start', 'end'])
            df['name'] = k
            df.at[1,'name']='TRIAL'
            if epochs is not None:
                epochs = epochs.append(df, ignore_index=True)
            else:
                epochs = df
                            
            current_time=epoch[0,1]
            
        avg_signal=signal.Signal(fs=fs, matrix=data, name=signal_to_average.name, 
                                 recording=signal_to_average.recording, 
                                 chans=signal_to_average.chans, epochs=epochs,  
                                 meta=signal_to_average.meta)
        newrec.add_signal(avg_signal)
        
        
    return newrec


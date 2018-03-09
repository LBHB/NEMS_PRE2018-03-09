import warnings
import numpy as np
import nems.epoch as ep
import pandas as pd
import nems.signal as signal

def generate_average_sig(signal_to_average,
                         new_signalname='respavg', epoch_regex='^STIM_'):
    '''
    Returns a signal with a new signal created by replacing every epoch
    matched in "epoch_regex" with the average of every occurrence in that
    epoch. This is often used to make a response average signal that
    is the same length as the original signal_to_average, usually for plotting.

    Optional arguments:
       signal_to_average   The signal from which you want to create an
                           average signal. It will not be modified.
       new_signalname      The name of the new, average signal.
       epoch_regex         A regex to match which epochs to average across.
    '''

    # 1. Fold matrix over all stimuli, returning a dict where keys are stimuli
    #    and each value in the dictionary is (reps X cell X bins)
    epochs_to_extract = ep.epoch_names_matching(signal_to_average.epochs, epoch_regex)
    folded_matrices = signal_to_average.extract_epochs(epochs_to_extract)

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0)

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg = signal_to_average.replace_epochs(per_stim_psth)
    respavg.name = new_signalname

    return respavg


def add_average_sig(rec, signal_to_average='resp',
                    new_signalname='respavg', epoch_regex='^STIM_'):
    '''
    Returns a recording with a new signal created by replacing every epoch
    matched in "epoch_regex" with the average of every occurrence in that
    epoch. This is often used to make a response average signal that
    is the same length as the original signal_to_average, usually for plotting.

    Optional arguments:
       signal_to_average   The signal from which you want to create an
                           average signal. It will not be modified.
       new_signalname      The name of the new, average signal.
       epoch_regex         A regex to match which epochs to average across.
    '''

    # generate the new signal by averaging epochs of the input singals
    respavg = generate_average_sig(rec[signal_to_average],
                                   new_signalname, epoch_regex)

    # Add the signal to the recording
    newrec = rec.copy()
    newrec.add_signal(respavg)

    return newrec


def average_away_epoch_occurrences(rec, epoch_regex='^STIM_'):
    '''
    Returns a recording with _all_ signals averaged across epochs that
    match epoch_regex, shortening them so that each epoch occurs only
    once in the new signals. i.e. unlike 'add_average_sig', the new
    recording will have signals 3x shorter if there are 3 occurrences of
    every epoch.

    This has advantages:
    1. Averaging the value of a signal (such as a response) in different
       occurrences will make it behave more like a linear variable with
       gaussian noise, which is advantageous in many circumstances.
    2. There will be less computation needed because the signal is shorter.

    It also has disadvantages:
    1. Stateful filters (FIR, IIR) will be subtly wrong near epoch boundaries
    2. Any ordering of epochs is essentially lost, unless all epochs appear
       in a perfectly repeated order.

    To avoid accidentally averaging away differences in responses to stimuli
    that are based on behavioral state, you may need to create new epochs
    (based on stimulus and behaviorial state, for example) and then match
    the epoch_regex to those.
    '''

    # Create new recording
    newrec = rec.copy()

    counter=0
    
    # iterate through each signal
    for signal_name, signal_to_average in rec.signals.items():
        # 1. Find matching epochs
        epochs_to_extract = ep.epoch_names_matching(rec.epochs, epoch_regex)

        # 2. Fold over all stimuli, returning a dict where keys are stimuli
        #    and each value in the dictionary is (reps X cell X bins)
        folded_matrices = signal_to_average.extract_epochs(epochs_to_extract)

        # force a standard list of sorted keys for all signals
        if counter==0:
            sorted_keys=list(folded_matrices.keys())
            sorted_keys.sort()
        counter+=1
        
        # 3. Average over all occurrences of each epoch, and append to data
        fs = signal_to_average.fs
        data = np.zeros([signal_to_average.nchans, 0])
        current_time = 0
        epochs = None
        for k in sorted_keys:
            # Supress warnings about all-nan matrices
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                per_stim_psth = np.nanmean(folded_matrices[k], axis=0)
            data = np.concatenate((data, per_stim_psth), axis=1)
            epoch = current_time+np.array([[0, per_stim_psth.shape[1]/fs]])
            df = pd.DataFrame(np.tile(epoch, [2, 1]), columns=['start', 'end'])
            df['name'] = k
            df.at[1, 'name'] = 'TRIAL'
            if epochs is not None:
                epochs = epochs.append(df, ignore_index=True)
            else:
                epochs = df
            current_time = epoch[0, 1]
            #print("{0} epoch: {1}-{2}".format(k,epoch[0,0],epoch[0,1]))
            
        avg_signal = signal.Signal(fs=fs, matrix=data,
                                   name=signal_to_average.name,
                                   recording=signal_to_average.recording,
                                   chans=signal_to_average.chans,
                                   epochs=epochs,
                                   meta=signal_to_average.meta)
        newrec.add_signal(avg_signal)

    return newrec

def generate_psth_from_est_for_both_est_and_val(est, val):
    '''
    Estimates a PSTH from the EST set, and returns two signals based on the
    est and val, in which each repetition of a stim uses the EST PSTH?
    '''
    # Method #0: Try to guess which stimuli have the most reps, use those for val
    est = rec.jackknife_by_time(10, 1, invert=False, excise=False)
    val = rec.jackknife_by_time(10, 1, invert=True, excise=False)

    epoch_regex='^STIM_'
    resp_est=est['resp']
    resp_val=val['resp']

    epochs_to_extract = ep.epoch_names_matching(resp_est.epochs, epoch_regex)
    folded_matrices = resp_est.extract_epochs(epochs_to_extract)

    # 2. Average over all reps of each stim and save into dict called psth.
    per_stim_psth = dict()
    for k in folded_matrices.keys():
        per_stim_psth[k] = np.nanmean(folded_matrices[k], axis=0)

    # 3. Invert the folding to unwrap the psth into a predicted spike_dict by
    #   replacing all epochs in the signal with their average (psth)
    respavg_est = resp_est.replace_epochs(per_stim_psth)
    respavg_est.name = 'stim'  # TODO: SVD suggests rename 2018-03-08
    est.add_signal(respavg_est)

    respavg_val = resp_val.replace_epochs(per_stim_psth)
    respavg_val.name = 'stim' # TODO: SVD suggests rename 2018-03-08
    val.add_signal(respavg_val)

    return est, val


def make_state_signal(rec, state_signals=['pupil'], permute_signals=[], new_signalname='state'):
    
    # TODO support for signals_permute
    if len(permute_signals):
        raise ValueError("permute_signals not yet supported") 

    x = np.ones([1,rec[state_signals[0]]._matrix.shape[1]])  # Much faster; TODO: Test if throws warnings
    ones_sig = rec[state_signals[0]]._modified_copy(x)
    ones_sig.name="baseline"
    
    state=signal.Signal.concatenate_channels([ones_sig]+[rec[x] for x in state_signals])
    state.name=new_signalname
    newrec = rec.copy()
    
    newrec.add_signal(state)

    return newrec


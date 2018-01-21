import os
import re
import json
import filecmp
import pytest
import pandas as pd
import numpy as np
from nems.signal import Signal

@pytest.fixture(scope='module')
def signal(signal_name='dummy_signal', recording_name='dummy_recording', fs=50,
           nchans=3, ntimes=200, ):
    '''
    Generates a dummy signal with a predictable structure (every element
    increases by 1) that is useful for testing.
    '''
    # Generate a numpy array that's incrementially increasing across channels,
    # then across timepoints, by 1.
    c = np.arange(nchans, dtype=np.float)
    t = np.arange(ntimes, dtype=np.float)
    data = c[..., np.newaxis] + t*nchans

    kwargs = {
        'matrix': data,
        'name': signal_name,
        'recording': recording_name,
        'chans': ['chan' + str(n) for n in range(nchans)],
        'fs': fs,
        'meta': {
            'for_testing': True,
            'date': "2018-01-10",
            'animal': "Donkey Hotey",
            'windmills': 'tilting'
            },
        'epochs': pd.DataFrame({'start_index': [0, 100, 150],
                       'end_index': [100, 150, 200],
                       'epoch_name': ['trial1', 'trial2', 'trial3']},
                        columns=['start_index', 'end_index', 'epoch_name'])
        }
    return Signal(**kwargs)


@pytest.fixture(scope='module')
def signal_tmpdir(tmpdir_factory):
    '''
    Test that signals object load/save methods work as intended, and
    return an example signal object for other tests.
    '''
    return tmpdir_factory.mktemp(__name__ + '_signal')

###################################################################
# TODO: new tests below. when finished tinkering, remove prints and
#       copy paste to test_signal.py

def test_fold_by_trial(signal):
    result = signal.fold_by('trial')
    assert result.shape == (3, 3, 100)

    # remove below later
    print("fold by trial: success")
    return result

def test_trial_epochs_from_reps(signal):
    cached_epochs = signal.epochs
    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    result1 = signal.fold_by('trial')
    assert result1.shape == (10, 3, 20)

    # remove later
    print("trial_epochs_from_reps (even): success")

    signal.epochs = signal.trial_epochs_from_reps(nreps=11)
    result2 = signal.fold_by('trial')
    assert result2.shape == (12, 3, 18)
    assert np.isnan(result2[11, 0]).sum() == 16

    # remove later
    print("trial_epochs_from_reps (uneven): success")
    signal.epochs = cached_epochs
    return result1, result2

def test_as_trials(signal):
    # TODO: need to decide if epochs have to be specified by user
    #       or if signal.as_trials() should grab defaults if no epochs present
    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    result = signal.as_trials()
    assert result.shape == (10, 3, 20)

def test_fold_by_pupil(signal):
    cached_epochs = signal.epochs
    pupil_info = pd.DataFrame({'start_index': [0, 150],
                               'end_index': [60, 190],
                               'epoch_name': ['pupil_closed1', 'pupil_closed2']
                               }, columns=['start_index', 'end_index',
                                           'epoch_name'])
    signal.epochs = signal.epochs.append(pupil_info, ignore_index=True)
    result = signal.fold_by('^pupil')
    assert result.shape == (2, 3, 60)

    # remove below later
    print("fold by pupil: success")
    signal.epochs = cached_epochs
    return result

def test_split_at_epoch(signal):
    cached_epochs = signal.epochs
    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    s1, s2 = signal.split_at_epoch(0.75)
    assert s1._matrix.shape == (3, 140)
    assert s2._matrix.shape == (3, 60)
    assert len(s1.epochs['epoch_name']) == 7
    assert len(s2.epochs['epoch_name']) == 3

    # remove below later
    print("split at epoch: success")
    signal.epochs = cached_epochs
    return s1, s2

"""
def test_fold_by_trial_and_pupil(signal):
    # TODO: not going to work yet b/c current implementation of fold_by
    #       would cause the slices to exceed index for multiple matches
    cached_epochs = signal.epochs
    pupil_info = pd.DataFrame({'start_index': [0, 150],
                               'end_index': [60, 190],
                               'epoch_name': ['pupil_closed1', 'pupil_closed2']
                               }, columns=['start_index', 'end_index',
                                           'epoch_name'])
    signal.epochs = signal.epochs.append(pupil_info, ignore_index=True)
    result = signal.fold_by('((^|, )(trial|stim))+$')
    assert result.shape == (5, 3, 60)

    # remove below later
    print("fold by both pupil and trial: success")
    signal.epochs = cached_epochs
    return result

s = signal()
r1 = test_fold_by_trial(s)
r2 = test_fold_by_pupil(s)
r3 = test_fold_by_trial_and_pupil(s)


s = signal()
matched_rows = s.epochs['epoch_name'].str.contains('trial8')
matched_epochs = s.epochs[matched_rows]
samples = matched_epochs['end_index'] - matched_epochs['start_index']
n_epochs = len(matched_epochs)
n_channels = s._matrix.shape[0]
n_samples = samples.max()

data = np.full((n_epochs, n_channels, n_samples), np.nan)
for i, (_, row) in enumerate(matched_epochs.iterrows()):
    lb, ub = row[['start_index', 'end_index']].astype('i')
    samples = ub-lb
    data[i, :, :samples] = s._matrix[:, lb:ub]

"""

s = signal()

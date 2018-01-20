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
    result = signal.fold_by('^trial')
    assert result.shape == (3, 3, 100)

    # remove below later
    print("fold by trial: success")
    return result

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

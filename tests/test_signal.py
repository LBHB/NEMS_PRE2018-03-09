import os
import json
import filecmp
import pytest
import numpy as np
import pandas as pd
import nems.signal
from nems.signal import Signal


@pytest.fixture(scope='module')
def signal(signal_name='dummy_signal', recording_name='dummy_recording', fs=50,
           nchans=3, ntimes=200, nreps=10):
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
        'epochs': pd.DataFrame({'start_index': [3, 15, 150],
                                'end_index': [200, 60, 190],
                                'epoch_name': ['trial1',
                                               'pupil_closed',
                                               'pupil_closed']},
                               columns=['start_index', 'end_index',
                                        'epoch_name']),
        'fs': fs,
        'meta': {
            'for_testing': True,
            'date': "2018-01-10",
            'animal': "Donkey Hotey",
            'windmills': 'tilting'
        },
    }
    return Signal(**kwargs)


@pytest.fixture(scope='module')
def signal_tmpdir(tmpdir_factory):
    '''
    Test that signals object load/save methods work as intended, and
    return an example signal object for other tests.
    '''
    return tmpdir_factory.mktemp(__name__ + '_signal')


def test_signal_save_load(signal, signal_tmpdir):
    '''
    Test that signals save and load properly
    '''
#    if not os.path.exists(signal_tmpdir):
#        os.mkdir(signal_tmpdir)    
    signal.save(str(signal_tmpdir), fmt='%1.3e')

    signals_found = Signal.list_signals(str(signal_tmpdir))
    assert len(signals_found) == 1

    save_directory = os.path.join(str(signal_tmpdir), signals_found[0])
    signal_loaded = Signal.load(save_directory)
    assert np.all(signal._matrix == signal_loaded._matrix)

    # TODO: add a test for the various signal attributes


def test_as_continuous(signal):
    assert signal.as_continuous().shape == (3, 200)


# TODO: Ivar broke this by changing the signal.epochs testing data structure.
#def test_fold_by(signal):
#    cached_epochs = signal.epochs
#    result = signal.fold_by('pupil')
#    assert result.shape == (2, 3, 60)
#    signal.epochs = cached_epochs


def test_trial_epochs_from_reps(signal):
    cached_epochs = signal.epochs
    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    result1 = signal.fold_by('trial')
    assert result1.shape == (10, 3, 20)

    signal.epochs = signal.trial_epochs_from_reps(nreps=11)
    result2 = signal.fold_by('trial')
    assert result2.shape == (12, 3, 18)
    assert np.isnan(result2[11, 0]).sum() == 16

    signal.epochs = cached_epochs  # Revert for next test


def test_as_trials(signal):
    cached_epochs = signal.epochs

    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    result = signal.as_trials()
    assert result.shape == (10, 3, 20)

    signal.epochs = cached_epochs  # Revert for next test


def test_as_average_trial(signal):
    cached_epochs = signal.epochs

    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    result = signal.as_average_trial()
    assert result.shape == (3, 20)

    signal.epochs = cached_epochs  # Revert for next test


def test_normalized_by_mean(signal):
    normalized_signal = signal.normalized_by_mean()
    data = normalized_signal.as_continuous()
    assert np.all(np.mean(data, axis=-1) == 0.0)
    assert np.allclose(np.std(data, axis=-1), 1.0)


def test_normalized_by_bounds(signal):
    normalized_signal = signal.normalized_by_bounds()
    data = normalized_signal.as_continuous()
    assert np.all(np.max(data, axis=-1) == 1)
    assert np.all(np.min(data, axis=-1) == -1)

"""
Replaced by ''_at_epoch, kept test temporarily for reference.
def test_split_at_rep(signal):
    left_signal, right_signal = signal.split_at_rep(0.8)
    assert left_signal.as_trials().shape == (8, 3, 20)
    assert right_signal.as_trials().shape == (2, 3, 20)
"""


def test_split_at_epoch(signal):
    cached_epochs = signal.epochs
    # set epochs = trial 0 - trial 9, length 20 each
    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    s1, s2 = signal.split_at_epoch(0.75)
    assert s1._matrix.shape == (3, 140)
    assert s2._matrix.shape == (3, 60)
    assert len(s1.epochs['epoch_name']) == 7
    assert len(s2.epochs['epoch_name']) == 3

    # add some extra epochs that overlap with the existing trial epochs
    overlapping_epochs = pd.DataFrame(
            {'start_index': [30, 70, 130], 'end_index': [65, 110, 180],
             'epoch_name': ['pupil1', 'pupil2', 'pupil3']},
            columns=['start_index', 'end_index', 'epoch_name']
            )

    signal.epochs = signal.epochs.append(overlapping_epochs, ignore_index=True)
    s3, s4 = signal.split_at_epoch(0.75)
    assert s3._matrix.shape == (3, 140)
    assert s4._matrix.shape == (3, 60)
    assert len(s3.epochs['epoch_name']) == 10
    assert len(s4.epochs['epoch_name']) == 4

    # TODO: @Ivar
    # to match previous functionality (which automatically reshaped data
    # to be rep x chan x time after split), just have to call as_trials
    # afterward. Leaving separate for now incase want to be able to to do both.
    m1 = s1.fold_by('trial')
    m2 = s2.fold_by('trial')
    assert m1.shape == (7, 3, 20)
    assert m2.shape == (3, 3, 20)

    signal.epochs = cached_epochs  # revert epochs for other tests


def test_split_at_time(signal):
    l, r = signal.split_at_time(0.81)
    print(signal.as_continuous().shape)
    assert l.as_continuous().shape == (3, 162)
    assert r.as_continuous().shape == (3, 38)

"""
Replaced by ''_by_epochs, kept test temporarily for reference.
def test_jackknifed_by_reps(signal):
    jsig = signal.jackknifed_by_reps(5, 1)
    isig = signal.jackknifed_by_reps(5, 1, invert=True)
    jdata = jsig.as_continuous()
    idata = isig.as_continuous()

    assert jdata.shape == (3, 200)
    assert idata.shape == (3, 200)

    assert np.sum(np.isnan(jdata)) == 120 # 3 channels x 1/5 * 200
    assert np.sum(np.isnan(idata)) == 480 # 3 channels x 4/5 * 200

    #assert(120 == np.sum(np.isnan(jsig.as_single_trial())))  # 3chan x 1/5 * 200
    #assert(480 == np.sum(np.isnan(isig.as_single_trial())))  # 3chan * 4/5 * 200
"""


def test_jackknifed_by_epochs(signal):
    cached_epochs = signal.epochs
    # set epochs to trial0 - trial9, length 20 each
    signal.epochs = signal.trial_epochs_from_reps(nreps=10)
    s1 = signal.jackknifed_by_epochs(regex='trial5')
    assert s1._matrix.shape == (3, 200) # shape shouldn't change
    assert np.isnan(s1._matrix).sum() == 60 # 3 chans x 20 samples x 1 epoch

    s2 = signal.jackknifed_by_epochs(regex='^trial(5|7|9)$')
    assert np.isnan(s2._matrix).sum() == 180 # 3 chans x 20 samples x 3 epochs

    s3 = signal.jackknifed_by_epochs(regex='trial4', invert=True)
    assert np.isnan(s3._matrix).sum() == 540 # 3 chans x 20 samples x 9 epochs
    # revert epochs to not interfere with other tests
    signal.epochs = cached_epochs


def test_jackknifed_by_time(signal):
    jsig = signal.jackknifed_by_time(20, 2)
    isig = signal.jackknifed_by_time(20, 2, invert=True)

    jdata = jsig.as_continuous()
    idata = isig.as_continuous()
    assert jdata.shape == (3, 200)
    assert idata.shape == (3, 200)

    assert np.sum(np.isnan(jdata)) == 30
    assert np.sum(np.isnan(idata)) == 570


def test_concatenate_time(signal):
    sig1 = signal
    sig2 = sig1.jackknifed_by_time(20, 2)
    sig3 = Signal.concatenate_time([sig1, sig2])
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (3, 400)


def test_concatenate_channels(signal):
    sig1 = signal
    sig2 = sig1.jackknifed_by_time(20, 2)
    sig3 = Signal.concatenate_channels([sig1, sig2])
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (6, 200)


def test_indexes_of_trues():
    ary = np.array([True, False, True, False])
    print("test 0")
    assert([[0, 1], [2, 3]] == Signal.indexes_of_trues(ary))

    ary = np.array([False, False, False, True, True, False])
    print("test 1")
    assert([[3, 5]] == Signal.indexes_of_trues(ary))

    ary = np.array([True, True, True, False, False])
    print("test 2")
    assert([[0, 3]] == Signal.indexes_of_trues(ary))

    ary = np.array([True, True, True, True, True, True])
    print("test 3")
    assert([[0, 6]] == Signal.indexes_of_trues(ary))

    ary = np.array([False, False, False])
    print("test 4")
    assert([] == Signal.indexes_of_trues(ary))

    ary = np.array([True, False, False])
    print("test 5")
    assert([[0, 1]] == Signal.indexes_of_trues(ary))

    ary = np.array([True])
    print("test 6")
    assert([[0, 1]] == Signal.indexes_of_trues(ary))


def test_resize_epoch(signal):
    df = signal.resize_epoch('pupil_closed', 3, 0, 'temp')
    assert([[12, 60, 'temp'], [147, 190, 'temp']] == df.values.tolist())

    df = signal.resize_epoch('pupil_closed', 0, 3, 'temp')
    assert([[15, 63, 'temp'], [150, 193, 'temp']] == df.values.tolist())


def test_combine_epochs(signal):
    print('Testing intersection...')
    df = signal.combine_epochs('pupil_closed', 'trial1', 'intersection', 'temp')
    assert([[15, 60, 'temp'], [150, 190, 'temp']] == df.values.tolist())
    print('Testing union...')
    df = signal.combine_epochs('pupil_closed', 'trial1', 'union', 'temp')
    assert([[3, 200, 'temp']] == df.values.tolist())
    print('Testing difference...')
    df = signal.combine_epochs('pupil_closed', 'trial1', 'difference', 'temp')
    assert([[3, 15, 'temp'], [60, 150, 'temp'], [190, 200, 'temp']] == df.values.tolist())


def test_overlapping_epochs(signal):
    print('Testing overlapping_epochs...')
    df = signal.overlapping_epochs('pupil_closed', 'trial1', 'temp')
    assert([[3, 200, 'temp']] == df.values.tolist())

import os
import json
import filecmp
import pytest
import numpy as np
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
        'fs': fs,
        'nreps': nreps,
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
    if not os.path.exists(signal_tmpdir):
        os.mkdir(signal_tmpdir)
    signal.save(signal_tmpdir, fmt='%1.3e')

    signals_found = Signal.list_signals(signal_tmpdir)
    assert len(signals_found) == 1

    save_directory = os.path.join(signal_tmpdir, signals_found[0])
    signal_loaded = Signal.load(save_directory)
    assert np.all(signal._matrix == signal_loaded._matrix)

    # TODO: add a test for the various signal attributes


def test_as_continuous(signal):
    assert signal.as_continuous().shape == (3, 200)


def test_as_average_trial(signal):
    result = signal.as_average_trial()
    assert result.shape == (3, 20)


def test_as_trials(signal):
    result = signal.as_trials()
    assert result.shape == (10, 3, 20)


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


def test_split_at_rep(signal):
    left_signal, right_signal = signal.split_at_rep(0.8)
    assert left_signal.as_trials().shape == (8, 3, 20)
    assert right_signal.as_trials().shape == (2, 3, 20)


def test_split_at_time(signal):
    l, r = signal.split_at_time(0.81)
    print(signal.as_continuous().shape)
    assert l.as_continuous().shape == (3, 162)
    assert r.as_continuous().shape == (3, 38)


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


def test_jackknifed_by_time(signal):
    jsig = signal.jackknifed_by_time(20, 2)
    isig = signal.jackknifed_by_time(20, 2, invert=True)

    jdata = jsig.as_continuous()
    idata = isig.as_continuous()
    assert jdata.shape == (3, 200)
    assert idata.shape == (3, 200)

    assert np.sum(np.isnan(jdata)) == 30
    assert np.sum(np.isnan(idata)) == 570


def test_append_signal(signal):
    sig1 = signal
    sig2 = sig1.jackknifed_by_time(20, 2)
    sig3 = sig1.append_signal(sig2)
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (3, 400)


def test_combine_channels(signal):
    sig1 = signal
    sig2 = sig1.jackknifed_by_time(20, 2)
    sig3 = sig1.combine_channels(sig2)
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (6, 200)

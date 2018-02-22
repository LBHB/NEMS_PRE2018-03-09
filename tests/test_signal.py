import os
import json
import filecmp
import pytest
import numpy as np
from numpy import nan
import pandas as pd
import nems.signal
from nems.signal import Signal, merge_selections


@pytest.fixture()
def signal(signal_name='dummy_signal', recording_name='dummy_recording', fs=50,
           nchans=3, ntimes=200):
    '''
    Generates a dummy signal with a predictable structure (every element
    increases by 1) that is useful for testing.
    '''
    # Generate a numpy array that's incrementially increasing across channels,
    # then across timepoints, by 1.
    c = np.arange(nchans, dtype=np.float)
    t = np.arange(ntimes, dtype=np.float)
    data = c[..., np.newaxis] + t*nchans

    epochs = pd.DataFrame({
        'start': [3, 15, 150],
        'end': [200, 60, 190],
        'name': ['trial', 'pupil_closed', 'pupil_closed']
        })
    epochs['start'] /= fs
    epochs['end'] /= fs
    kwargs = {
        'matrix': data,
        'name': signal_name,
        'recording': recording_name,
        'chans': ['chan' + str(n) for n in range(nchans)],
        'epochs': epochs,
        'fs': fs,
        'meta': {
            'for_testing': True,
            'date': "2018-01-10",
            'animal': "Donkey Hotey",
            'windmills': 'tilting'
        },
    }
    return Signal(**kwargs)


@pytest.fixture()
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

def test_epoch_save_load(signal, signal_tmpdir):
    '''
    Test that epochs save and load properly
    '''

    before = signal.epochs

    signal.save(str(signal_tmpdir), fmt='%1.3e')
    signals_found = Signal.list_signals(str(signal_tmpdir))
    save_directory = os.path.join(str(signal_tmpdir), signals_found[0])
    signal_loaded = Signal.load(save_directory)

    after = signal_loaded.epochs
    print("Dataframes equal?\n"
          "Before:\n{0}\n"
          "After:\n{1}\n"
          .format(before, after))
    assert before.equals(after)


def test_as_continuous(signal):
    assert signal.as_continuous().shape == (3, 200)


def test_extract_epoch(signal):
    result = signal.extract_epoch('pupil_closed')
    assert result.shape == (2, 3, 45)


def test_trial_epochs_from_occurrences(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=10)
    result1 = signal.extract_epoch('trial')
    assert result1.shape == (10, 3, 20)

    with pytest.raises(ValueError):
        signal.epochs = signal.trial_epochs_from_occurrences(occurrences=11)


def test_as_trials(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=10)
    result = signal.extract_epoch('trial')
    assert result.shape == (10, 3, 20)


def test_as_average_trial(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=10)
    result = signal.average_epoch('trial')
    assert result.shape == (3, 20)


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


def test_split_at_time(signal):
    l, r = signal.split_at_time(0.81)
    print(signal.as_continuous().shape)
    assert l.as_continuous().shape == (3, 162)
    assert r.as_continuous().shape == (3, 38)


def test_jackknife_by_epoch(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=50)
    s1 = signal.jackknife_by_epoch(10, 0, 'trial', tiled=False, invert=True)
    assert s1._matrix.shape == (3, 200)  # shape shouldn't change
    assert(1770.0 == np.nansum(s1.as_continuous()[:]))


def test_jackknife_by_time(signal):
    jsig = signal.jackknife_by_time(20, 2)
    isig = signal.jackknife_by_time(20, 2, invert=True)

    jdata = jsig.as_continuous()
    idata = isig.as_continuous()
    assert jdata.shape == (3, 200)
    assert idata.shape == (3, 200)

    assert np.sum(np.isnan(jdata)) == 30
    assert np.sum(np.isnan(idata)) == 570


def test_concatenate_time(signal):
    sig1 = signal
    sig2 = sig1.jackknife_by_time(20, 2)
    sig3 = Signal.concatenate_time([sig1, sig2])
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (3, 400)


def test_concatenate_channels(signal):
    sig1 = signal
    sig2 = sig1.jackknife_by_time(20, 2)
    sig3 = Signal.concatenate_channels([sig1, sig2])
    assert sig1.as_continuous().shape == (3, 200)
    assert sig3.as_continuous().shape == (6, 200)


def test_add_epoch(signal):
    epoch = np.array([[0, 200]])
    signal.add_epoch('experiment', epoch)
    assert len(signal.epochs) == 4
    assert np.all(signal.get_epoch_bounds('experiment') == epoch)


def test_merge_selections(signal):
    signals = []
    for i in range(10):
        jk = signal.jackknife_by_time(10, i, invert=True)
        signals.append(jk)

    merged = merge_selections(signals)

    # merged and signal should be identical
    assert np.sum(np.isnan(merged.as_continuous())) == 0
    assert np.array_equal(signal.as_continuous(), merged.as_continuous())
    assert signal.epochs.equals(merged.epochs)

    # This should not throw an exception
    merge_selections([signal, signal, signal])

    normalized = signal.normalized_by_mean()

    # This SHOULD throw an exception because they totally overlap
    with pytest.raises(ValueError):
        merge_selections([signal, normalized])

    jk2 = normalized.jackknife_by_time(10, 2, invert=True)
    jk3 = signal.jackknife_by_time(10, 3, invert=True)
    jk4 = signal.jackknife_by_time(10, 4, invert=True)

    # This will NOT throw an exception because they don't overlap
    merged = merge_selections([jk2, jk3])
    merged = merge_selections([jk2, jk4])

    # This SHOULD throw an exception
    with pytest.raises(ValueError):
        merged = merge_selections([signal, jk2])


def test_extract_channels(signal):
    two_sig = signal.extract_channels([0, 1])
    assert two_sig.shape == (2, 200)
    one_sig = signal.extract_channels(2)
    assert one_sig.shape == (1, 200)
    recombined = Signal.concatenate_channels([two_sig, one_sig])
    before = signal.as_continuous()
    after = recombined.as_continuous()
    assert np.array_equal(before, after)


def test_string_syntax_valid(signal):
    assert(signal._string_syntax_valid('this_is_fine'))
    assert(signal._string_syntax_valid('THIS_IS_FINE_TOO'))
    assert(not signal._string_syntax_valid('But this is not ok'))


def test_jackknifes_by_epoch(signal):
    signal.epochs = signal.trial_epochs_from_occurrences(occurrences=50)
    for est, val in signal.jackknifes_by_epoch(10, 'trial'):
        print(np.nansum(est.as_continuous()[:]),
              np.nansum(val.as_continuous()[:]),)
    # This is not much of a test -- I'm just running the generator fn!
    assert(True)

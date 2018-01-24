from .signal import Signal
from .recording import Recording


def recording_like(recording, signals):
    '''
    Make a copy of the recording and update it with the provided signals.

    Parameters
    ----------
    recording : instance of `Recording`
        Recording to copy
    signals : dict
        Mapping of signal name to Signal objects. These signals will be saved in
        the copy of the recording. Signals present in `recording` but not
        `signals` will appear in the copy of the recording.

    Examples
    --------
    Create a copy of the recording and overwrite the 'pred' signal.
    >>> old_pred = recording.get_signal('pred')
    >>> new_pred = transform_signal(old_pred)
    >>> new_recording = recording_like(recording, {'pred': new_pred})

    Note
    ----
    This only performs a shallow copy, so the copy of the recording object may
    point to a set of signal objects that are shared by the original recording.
    This will minimize memory consumption.

    The name of the function is meant to mimic Numpy's `full_like`,
    `empty_like`, `ones_like`, etc. functions as a mneumonic.
    '''
    new_recording = recording.copy()
    new_recording.set_signals(signals)
    return new_recording


def signal_like(signal, data, **kwargs):
    '''
    Make a copy of the signal and update it with the provided data.

    Parameters
    ----------
    signal : instance of `Signal`
        Signal to copy
    data : 2D array
        Array containing the data to store in the copy of the signal.

    Other arguments passed in by keyword can be any attribute on the Signal
    object that is settable via the `Signal.__init__` method.

    Examples
    --------
    >>> pred = recording.get_signal('pred')
    >>> x = pred.as_continuous()
    >>> y = evaluate(x)
    >>> new_pred = signal_like(pred, data=y)
    '''
    return signal._modified_copy(data, **kwargs)

from functools import partial
import numpy as np
import scipy.signal


def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    null_data = np.full(n_taps*2, x[0])
    zi = np.ones(n_taps-1)
    return scipy.signal.lfilter(b, [1], null_data, zi=zi)[1]


def _fir_filter(x, coefficients):
    '''
    Private function used by fir_filter().
    '''
    result = []
    for x, c in zip(x, coefficients):
        zi = get_zi(c, x)
        r, zf = scipy.signal.lfilter(c, [1], x, zi=zi)
        result.append(r[np.newaxis])
    result = np.concatenate(result)
    return np.sum(result, axis=-2, keepdims=True)


def fir_filter(rec, i, o, coefficients):
    # fn = lambda x: _fir_filter(x, coefficients)
    fn = partial(_fir_filter, coefficients=coefficients)
    return [rec[i].transform(fn).rename(o)]

import numpy as np


def sum_channels(rec=None, i=None, o=None):
    '''
    (NaN-)sums all the channels together.
    '''
    fn = lambda x: np.nansum(x, axis=0, keepdims=True)
    return rec.add_signal(rec[i].transform(fn, o))

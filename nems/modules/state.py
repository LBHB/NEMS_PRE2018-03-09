import numpy as np

def state_dc_gain(rec, i, g, d, o):
    '''
    Parameters
    ----------
    i name of input
    g - gain to scale by
    d - dc to offset by
    o name of output signal
    '''
    
    fn = lambda x: rec[g]._matrix * x + rec[d]._matrix
    return [rec[i].transform(fn, o)]


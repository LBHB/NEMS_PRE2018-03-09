"""
modules/state.py

functions for applying state-related transformations
"""

#import numpy as np

def state_dc_gain(rec, i, o, s, g, d):
    '''
    Parameters
    ----------
    i name of input
    o name of output signal
    s name of state signal
    g - gain to scale s by
    d - dc to offset by
    '''
    
    fn = lambda x: g @ rec[s]._matrix * x + d @ rec[s]._matrix
    
    return [rec[i].transform(fn, o)]


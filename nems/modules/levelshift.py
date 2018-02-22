import numpy as np
import cProfile

@profile
def levelshift(rec, i, o, level):
    '''
    Parameters
    ----------
    level : a scalar to add to every element of the input signal.
    '''
    fn = lambda x: x + level
    return [rec[i].transform(fn, o)]

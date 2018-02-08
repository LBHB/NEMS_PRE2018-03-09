import numpy as np


def weight_channels(rec=None, i=None, o=None, coefficients=None):
    '''
    Parameters
    ----------
    coefficients : 2d array (output channel x input channel weights)
        Weighting of the input channels. A set of weights are provided for each
        desired output channel. Each row in the array are the weights for the
        input channels for that given output. The length of the row must be
        equal to the number of channels in the input array
        (e.g., `x.shape[-3] == coefficients.shape[-1]`).
    '''
    fn = lambda x: coefficients @ x
    return [rec[i].transform(fn, o)]

from functools import partial


def _weight_channels(x, coefficients):
    return coefficients @ x


def weight_channels(rec, i, o, coefficients):
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
    # fn = lambda x: coefficients @ x
    #fn = partial(_weight_channels, coefficients=coefficients)
    #return [rec[i].transform(_double_exponential,
    #                         args=(base, amplitude, shift, kappa)).rename(o)]
    return [rec[i].transform(_weight_channels,
                             (coefficients)).rename(o)]

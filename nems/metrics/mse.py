import numpy as np


def mse(result, pred_name='pred', resp_name='resp'):
    '''
    Given the evaluated data, return the mean squared error

    Parameters
    ----------
    result : dictionary of arrays
        Output of `model.evaluate(phi, data)`
    pred_name : string
        Name of prediction in the result dictionary
    resp_name : string
        Name of response in the result dictionary

    Returns
    -------
    mse : float
        Mean-squared difference between the prediction and response.

    Example
    -------
    >>> result = model.evaluate(data, phi)
    >>> error = mse(result, 'pred', 'resp')

    Note
    ----
    This function is written to be compatible with both numeric (i.e., Numpy)
    and symbolic (i.e., Theano, TensorFlow) computation. Please do not edit
    unless you know what you're doing.
    '''
    pred = result[pred_name]
    resp = result[resp_name]
    squared_errors = (pred-resp)**2
    return np.nanmean(squared_errors)

"""Defines evaluator functions that accept a modelspec and some data,
and output the same data with 'pred' updated.

Typically used as part of a cost function. Modelspec should already
contain the updated phi provided by the fitter when used in this manner.

"""

from . import api

def matrix_eval(data, modelspec):
    # TODO
    # Goal is to have each module be a 'pure' function that takes in
    # an ndarray and returns an ndarray, with no side-effects.

    d_in = data.signals['stim'].as_continuous()
    stack = [d_in]
    for m in modelspec:
        fn = getattr(api, m['api'])
        d_out = fn(d_in, **m['fn_kwargs'])
        stack.append(d_out.copy())
        d_in = d_out

    # TODO: Calling modified copy which is tagged as internal use, but
    #       seems like this is the best way to get matrix back to a
    #       pred signal (since other attributes of pred shouldn't change)
    resp = data.signals['resp']._modified_copy(stack[-1])
    data.signals['resp'] = resp
    return data

def signal_eval(data, modelspec):
    # Same as matrix_eval, but passes the signal object around instead of
    # just the data matrix.

    d_in = data.signals['stim'].copy()
    stack = [d_in]
    for m in modelspec:
        fn = getattr(api, m['api'])
        d_out = fn(d_in, **m['fn_kwargs'])
        stack.append(d_out.copy())
        d_in = d_out

    data.signals['pred'] = stack[-1]
    return data
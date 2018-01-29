"""Defines evaluator functions that accept a modelspec and some data,
and output the same data with 'pred' updated.

Typically used as part of a cost function. Modelspec should already
contain the updated phi provided by the fitter when used in this manner.

"""

from . import api

def matrix_eval(data, modelspec):
    d_in = data.get_signal('stim').as_continuous()
    stack = [d_in]
    for m in modelspec:
        fn = getattr(api, m['api'])
        phi = m['phi'].values()
        kwargs = m['fn_kwargs']
        d_out = fn(d_in, *phi, **kwargs)
        stack.append(d_out.copy())
        d_in = d_out

    # TODO: Calling modified copy which is tagged as internal use, but
    #       seems like this is the best way to get matrix back to a
    #       pred signal (since other attributes of pred shouldn't change).
    #       Using stim as workaround template for now, but maybe there's
    #       a smarter way to initialize a 'pred' signal?
    # Use stim signal as template for pred, since meta attributes should
    # be the same even though array values updated.
    pred = data.get_signal('stim')._modified_copy(stack[-1])
    data.set_signal('pred', pred)
    return data

def signal_eval(data, modelspec):
    # Same as matrix_eval, but passes the signal object around instead of
    # just the data matrix.

    d_in = data.get_signal('stim').copy()
    stack = [d_in]
    for m in modelspec:
        fn = getattr(api, m['api'])
        phi = m['phi'].values()
        kwargs = m['fn_kwargs']
        d_out = fn(d_in, *phi, **kwargs)
        stack.append(d_out.copy())
        d_in = d_out

    data.set_signal('pred', stack[-1])
    return data

def matrix_eval(data, modelspec):
    d_in = data.get_signal('stim').as_continuous()
    stack = [d_in]
    for m in modelspec:
        if m['fn'] in lookup_table:
            fn = lookup_table[m['fn']]
        else:
            api, fn_name = split_to_api_and_fn(m['fn'])
            api_obj = import_module(api)
            fn = getattr(api_obj, fn_name)
            lookup_table[m['fn']] = fn
        phi = m['phi'].values()
        kwargs = m['fn_kwargs']
        # d_out = fn(d_in, *phi, **kwargs)
        d_out = fn(d_in, *phi)
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
    pred.name = 'pred'
    data.add_signal(pred)
    return data

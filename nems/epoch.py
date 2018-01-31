import numpy as np


def epoch_union(a, b):
    '''
    Compute the union of the epochs.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.

    Returns
    -------
    union : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the union of a and b. Note that O <= M +
        N.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result: [    ]  [         ] []     [     ]
    '''
    epochs = np.concatenate((a, b), axis=0)
    epochs.sort(axis=0)
    i = 0
    n = len(epochs)
    union = []

    while i < n:
        lb, ub = epochs[i]
        i += 1
        while (i < n) and (ub >= epochs[i, 0]):
            ub = epochs[i, 1]
            i += 1
        union.append((lb, ub))
    return np.array(union)


def epoch_difference(a, b):
    '''
    Compute the difference of the epochs. All regions in a which overlap with b
    will be removed.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.

    Returns
    -------
    difference : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the difference of a and b.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result:     []  [  ]  [   ]        []
    '''
    a = a.tolist()
    a.sort(reverse=True)
    b = b.tolist()
    b.sort(reverse=True)

    difference = []
    lb, ub = a.pop()
    lb_b, ub_b = b.pop()

    while True:
        if lb > ub_b:
            #           [ a ]
            #     [ b ]
            # Current epoch in b ends before current epoch in a. Move onto
            # the next epoch in b.
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                difference.append((lb, ub))
                break
        elif ub <= lb_b:
            #   [  a    ]
            #               [ b        ]
            # Current epoch in a ends before current epoch in b. Add bounds
            # and move onto next epoch in a.
            difference.append((lb, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb == lb_b) and (ub == ub_b):
            try:
                lb, ub = a.pop()
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub > ub_b):
            #   [  a    ]
            #     [ b ]
            # Current epoch in b is fully contained in the  current epoch
            # from a. Save everything in
            # a up to the beginning of the current epoch of b. However, keep
            # the portion of the current epoch in a
            # that follows the end of the current epoch in b so we can
            # detremine whether there are additional epochs in b that need
            # to be cut out..
            difference.append((lb, lb_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                difference.append((lb, ub))
                break
        elif (lb <= lb_b) and (ub <= ub_b):
            #   [  a    ]
            #     [ b        ]
            # Current epoch in b begins in a, but extends past a.
            difference.append((lb, lb_b))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (ub > lb_b) and (lb <= ub_b):
            #   [  a    ]
            # [       b     ]
            # Current epoch in a is fully contained in b
            lb, ub = a.pop()
        elif (ub > lb_b) and (lb > ub_b):
            #   [  a    ]
            # [ b    ]
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                difference.append((lb, ub))
        else:
            # This should never happen.
            m = 'Unhandled epoch boundary condition. Contact the developers.'
            raise SystemError(m)

    # Add all remaining epochs from a
    difference.extend(a[::-1])
    return np.array(difference)


def epoch_intersection(a, b):
    '''
    Compute the intersection of the epochs. Only regions in a which overlap with
    b will be kept.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.

    Returns
    -------
    intersection : 2D array of (O x 2)
        The first column is the start time and second column is the end time. O
        is the number of occurances of the difference of a and b.

    Example
    -------
    a:       [   ]  [         ]        [ ]
    b:      [   ]       [ ]     []      [    ]
    result:  [  ]       [ ]             []
    '''
    # Convert to a list and then sort in reversed order such that pop() walks
    # through the occurences from earliest in time to latest in time.
    a = a.tolist()
    a.sort(reverse=True)
    b = b.tolist()
    b.sort(reverse=True)

    intersection = []
    lb, ub = a.pop()
    lb_b, ub_b = b.pop()

    while True:
        if lb > ub_b:
            #           [ a ]
            #     [ b ]
            # Current epoch in b ends before current epoch in a. Move onto
            # the next epoch in b.
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif ub <= lb_b:
            #   [  a    ]
            #               [ b        ]
            # Current epoch in a ends before current epoch in b. Add bounds
            # and move onto next epoch in a.
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (lb == lb_b) and (ub == ub_b):
            #   [  a    ]
            #   [  b    ]
            # Current epoch in a matches epoch in b.
            try:
                intersection.append((lb, ub))
                lb, ub = a.pop()
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub >= ub_b):
            #   [  a    ]
            #     [ b ]
            # Current epoch in b is fully contained in the  current epoch
            # from a. Save everything in
            # a up to the beginning of the current epoch of b. However, keep
            # the portion of the current epoch in a
            # that follows the end of the current epoch in b so we can
            # detremine whether there are additional epochs in b that need
            # to be cut out..
            intersection.append((lb_b, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        elif (lb <= lb_b) and (ub <= ub_b):
            #   [  a    ]
            #     [ b        ]
            # Current epoch in b begins in a, but extends past a.
            intersection.append((lb_b, ub))
            try:
                lb, ub = a.pop()
            except IndexError:
                break
        elif (ub > lb_b) and (lb <= ub_b):
            #   [  a    ]
            # [       b     ]
            # Current epoch in a is fully contained in b
            intersection.append((lb, ub))
            lb, ub = a.pop()
        elif (ub > lb_b) and (lb > ub_b):
            #   [  a    ]
            # [ b    ]
            intersection.append((lb, ub_b))
            lb = ub_b
            try:
                lb_b, ub_b = b.pop()
            except IndexError:
                break
        else:
            # This should never happen.
            m = 'Unhandled epoch boundary condition. Contact the developers.'
            raise SystemError(m)

    # Add all remaining epochs from a
    intersection.extend(a[::-1])
    return np.array(intersection)


def _epoch_contains_mask(a, b):
    mask = [(b >= lb) & (b < ub) for lb, ub in a]
    return np.concatenate([m[np.newaxis] for m in mask], axis=0)


def epoch_contains(a, b, mode):
    '''
    Tests whether an occurence of a contains an occurence of b.

    Parameters
    ----------
    a : 2D array of (M x 2)
        The first column is the start time and second column is the end time. M
        is the number of occurances of a.
    b : 2D array of (N x 2)
        The first column is the start time and second column is the end time. N
        is the number of occurances of b.
    mode : {'start', 'end', 'both', 'any'}
        Test to perform.
        - 'start' requires only the start of b to be contained in a
        - 'end' requires only the end of b to be contained in a
        - 'both' requires both start and end in b to be contained in a
        - 'any' is True anywhere b partially or completely overlaps with a

    Returns
    -------
    mask : 1D array of len(a)
        Boolean mask indicating whether the corresponding entry in a meets the
        test criteria.
    '''
    mask = _epoch_contains_mask(a, b)
    if mode == 'start':
        return mask[:, :, 0].any(axis=1)
    elif mode == 'end':
        return mask[:, :, 1].any(axis=1)
    elif mode == 'both':
        return mask.all(axis=2).any(axis=1)
    elif mode == 'any':
        b_in_a = mask.any(axis=2).any(axis=1)
        # This mask will not capture situations where an occurence of a is fully
        # contained in an occurence of b. To test for this, we can flip the
        # epochs and build a new mask to perform this special-case test.
        mask = _epoch_contains_mask(b, a)
        a_in_b = mask.any(axis=2).any(axis=0)
        return b_in_a | a_in_b

def phi_to_vector(phi):
    '''
    Convert a list of dictionaries where the values are scalars or array-like
    to a single vector.

    This is a helper function for fitters that use scipy.optimize. The scipy
    optimizers require phi to be a single vector; however, it's more intuitive
    for us to return a list of dictionaries (where each dictionary in the list
    contains the values to be fitted.

    >>> phi = [{'baseline': 0, 'coefs': [32, 41]}, {}, {'a': 1, 'b': 15.1}]
    >>> phi_to_vector(phi)
    [0, 32, 41, 1, 15.1]

    >>> phi = [{'coefs': [[1, 2], [3, 4], [5, 6]]}, {'a': 32}]
    >>> phi_to_vector(phi)
    [1, 2, 3, 4, 5, 6, 32]
    '''
    vector = []
    for p in phi:
        for k in sorted(p.keys()):
            value = p[k]
            if np.isscalar(value):
                vector.append(value)
            else:
                flattened_value = np.asanyarray(value).ravel()
                vector.extend(flattened_value)
    return vector


def vector_to_phi(vector, phi_template):
    '''
    Convert vector back to a list of dictionaries given a template for phi

    >>> phi_template = [{'baseline': 0, 'coefs': [0, 0]}, {}, {'a': 0, 'b': 0}]
    >>> vector = [0, 32, 41, 1, 15.1]
    >>> vector_to_phi(vector, phi_template)
    [{'baseline': 0, 'coefs': array([32, 41])}, {}, {'a': 1, 'b': 15.1}]

    >>> phi_template = [{'coefs': [[0, 0], [0, 0], [0, 0]]}, {'a': 0}]
    >>> vector = [1, 2, 3, 4, 5, 6, 32]
    >>> vector_to_phi(vector, phi_template)
    [{'coefs': array([[1, 2], [3, 4], [5, 6]])}, {'a': 32}]
    '''
    # TODO: move this to a unit test instead?
    offset = 0
    phi = []
    for p_template in phi_template:
        p = {}
        for k in sorted(p_template.keys()):
            value_template = p_template[k]
            if np.isscalar(value_template):
                value = vector[offset]
                offset += 1
            else:
                value_template = np.asarray(value_template)
                size = value_template.size
                value = np.asarray(vector[offset:offset+size])
                value.shape = value_template.shape
                offset += size
            p[k] = value
        phi.append(p)
    return phi

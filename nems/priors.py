import nems.keywords
import nems.distributions.api

# For creating values of 'phi' from the priors in a modelspec.
#
# The three important publicly used functions are:
#    set_mean_phi
#    set_random_phi
#    set_percentile_phi
#
# Note: A 'prior' in a modelspec is a dict of distributions, where keys
# are parameter names and values are tuples used to instantiate the
# nems/distribution.py objects.


def _get_module_prior(module, default_priors=nems.keywords.default_priors):
    '''
    Returns the prior distributions for a module. If a 'prior' field
    exists in module, its distributions are used to replace any default
    priors listed in keywords/default_priors.
    '''
    if module['fn'] in default_priors:
        general_prior = default_priors[module['fn']]
        if 'prior' in module:
            specific_prior = module['prior']
            prior = {**general_prior, **specific_prior}
        else:
            prior = general_prior
        # Now instantiate the prior distribution objects
        for arg_name in prior:
            dist, dist_params = prior[arg_name]
            dist_class = getattr(nems.distributions.api, dist)
            dist = dist_class(*dist_params)
            prior[arg_name] = dist
    else:
        prior = None

    return prior


def _to_phi(prior, method='mean', percentile=50):
    '''
    Not for public use.
    Sample from a single module prior; method must be 'mean', 'sample', or
    'percentile'. Returns a dict of parameters suitable for use as 'phi'.
    '''
    phi = {}
    for param_name in prior:
        dist = prior[param_name]
        if method is 'mean':
            ary = dist.mean()
        elif method is 'sample':
            ary = dist.sample()
        elif method is 'percentile':
            ary = dist.percentile(percentile)
        else:
            raise ValueError('_to_phi got invalid method name.')
        print(ary.shape)
        phi[param_name] = ary.tolist()
    return phi


def _add_phi_to_module(module, prior_to_phi_fn):
    '''
    Not for public use.
    Returns a module identical to the one provided, but with phi
    initalized using the priors, if they exist but phi does not.
    '''
    new_module = module.copy()
    prior = _get_module_prior(module)
    if prior:
        phi_from_prior = prior_to_phi_fn(prior)
        if 'phi' in new_module:
            phi_from_modelspec = new_module['phi']
            new_phi = {**phi_from_prior, **phi_from_modelspec}
            new_module['phi'] = new_phi
        else:
            new_module['phi'] = phi_from_prior
    return new_module


def _add_phi_to_modelspec(modelspec, prior_to_phi_fn):
    '''
    Not for public use.
    Initializes phi for each module in the modelspec, if one does not
    already exist for each module. Optional arguments start_idx and
    stop_idx let you initialize only certain modules in the modelspec.
    '''
    new_mspec = [_add_phi_to_module(m, prior_to_phi_fn) for m in modelspec]
    return new_mspec


def set_mean_phi(modelspec):
    '''
    Returns a modelspec with phi set to the expected values of priors.
    '''
    prior_to_phi_fn = lambda prior: _to_phi(prior, 'mean')
    return _add_phi_to_modelspec(modelspec, prior_to_phi_fn)


def set_random_phi(modelspec):
    '''
    Returns a modelspec with phi that are random samples from the priors.
    '''
    prior_to_phi_fn = lambda prior: _to_phi(prior, 'sample')
    return _add_phi_to_modelspec(modelspec, prior_to_phi_fn)


def set_percentile_phi(modelspec, percentile):
    '''
    Returns a modelspec with phi that falls at the given percentile of the
    Cumulative Density Function (CDF) for each module's parameters.

    Note: Because our priors are not a joint probability distribution,
    if you have 2 independent parameters and use the 10% percentile, this
    means the 10% percentile on each one, giving you the 0.10 * 0.10 = 0.01
    or 1% percentile joint prob sample. In other words, this is fine for
    bounds plotting or CI, but if you really want the joint distribution
    percentile, CDF you will need to rescale percentile accordingly. TODO.
    '''
    prior_to_phi_fn = lambda prior: _to_phi(prior, 'percentile', percentile)
    return _add_phi_to_modelspec(modelspec, prior_to_phi_fn)

import logging as log
from nems.distributions.distribution import Distribution
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

# If not otherwise specified, the following default priors will be used as
# a last resort to initialize a module. They do so based only upon the function
# name rather than a keyword, and so must be 1x1 distributions which can be
# broadcast to match the array size that is needed.
default_priors = {'nems.modules.fir.fir_filter':
                  {'coefficients': ('Normal', {'mu': [0], 'sd': [1]})},

                  'nems.modules.levelshift.levelshift':
                  {'level': ('Normal', {'mu': [0], 'sd': [10]})},

                  'nems.modules.nonlinearity.double_exponential':
                  {'base': ('Normal', {'mu': [0], 'sd': [1]}),
                   'amplitude': ('Normal', {'mu': [0.2], 'sd': [1]}),
                   'shift': ('Normal', {'mu': [0], 'sd': [1]}),
                   'kappa': ('Normal', {'mu': [0], 'sd': [0.1]})}}
# TODO: I don't like that this is here. Where to put it? -- Ivar


def _get_module_prior(module, default_priors=default_priors):
    '''
    Returns the prior distributions for a module. If a 'prior' field
    exists in module, its distributions are used to replace any default
    priors listed in keywords/default_priors.
    '''
    specific_prior = module.get('prior')
    general_prior = default_priors.get(module['fn'])

    if general_prior and specific_prior:
        prior = {**general_prior, **specific_prior}
    elif specific_prior:
        prior = specific_prior
    elif general_prior:
        prior = general_prior
    else:
        return None

    # Instantiate the Distribution objects if needed.
    for arg_name in prior:
        if issubclass(type(prior[arg_name]), Distribution):
            # Skip this; it is already instantiated
            continue
        # TODO: this is resulting in error some times, due to
        #       the following code being called on a Normal distribution
        #       object. So the if(issubclass...) condition isn't catching
        #       some or all of the distribution objects.
        dist_type, dist_params = prior[arg_name]
        dist_class = getattr(nems.distributions.api, dist_type)
        dist = dist_class(**dist_params)
        # log.debug(arg_name, dist_type, dist_params, dist)
        prior[arg_name] = dist

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
        phi[param_name] = ary.tolist()
    return phi


def _set_phi_in_module(module, prior_to_phi_fn):
    '''
    Not for public use.
    Returns a module identical to the one provided, but with phi
    initalized using the priors.
    '''
    new_module = module.copy()
    prior = _get_module_prior(module)
    if not prior:
        if 'phi' in new_module:
            m = 'Phi exists w/o prior: ' + str(module)
            log.warn(m)
    else:
        new_phi = prior_to_phi_fn(prior)
        if 'phi' in new_module:
            old_phi = new_module['phi']
            if len(new_phi) != len(old_phi):
                m = 'Not all phi values have priors: ' + str(module)
                log.warn(m)
            new_module['phi'] = {**old_phi, **new_phi}
        else:
            new_module['phi'] = new_phi
    return new_module


def _set_phi_in_modelspec(modelspec, prior_to_phi_fn):
    '''
    Not for public use.
    Initializes phi for each module in the modelspec, if one does not
    already exist for each module.
    '''
    new_mspec = [_set_phi_in_module(m, prior_to_phi_fn) for m in modelspec]
    return new_mspec


def set_mean_phi(modelspec):
    '''
    Returns a modelspec with phi set to the expected values of priors.
    '''
    prior_to_phi_fn = lambda prior: _to_phi(prior, 'mean')
    return _set_phi_in_modelspec(modelspec, prior_to_phi_fn)


def set_random_phi(modelspec):
    '''
    Returns a modelspec with phi set to random samples from the priors.
    '''
    prior_to_phi_fn = lambda prior: _to_phi(prior, 'sample')
    return _set_phi_in_modelspec(modelspec, prior_to_phi_fn)


def set_percentile_phi(modelspec, percentile):
    '''
    Returns a modelspec with phi set to what falls at the given percentile of
    the Cumulative Density Function (CDF) of each module's parameters.

    Note: Because our priors are not a joint probability distribution,
    if you have 2 independent parameters and use the 10% percentile, this
    means the 10% percentile on each one, giving you the 0.10 * 0.10 = 0.01
    or 1% percentile joint prob sample. In other words, this is fine for
    bounds plotting or CI, but if you really want the joint distribution
    percentile, CDF you will need to rescale percentile accordingly. TODO.
    '''
    prior_to_phi_fn = lambda prior: _to_phi(prior, 'percentile', percentile)
    return _set_phi_in_modelspec(modelspec, prior_to_phi_fn)

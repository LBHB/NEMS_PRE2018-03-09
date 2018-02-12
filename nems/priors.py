from functools import partial
import nems.distributions.api as dist_api

# default mappings of transformation functions to priors

# TODO: Change these to reasonable values where necessary.
#       I wasn't sure which distributions/means/etc made
#       sense for which parameters.     -jacob  2/10/2018

def set_priors(modelspec, priors, i=0, j=-1):
    """
    Given a modelspec and a list of prior specifications, assigns
    each prior to the modelspec entry at the corresponding index.
    Kwargs i and j can be specified to set arbitrary
    start and end indices, respectively.

    modelspec : dict
        See nems/docs/modelspec.md

    priors : list
        Each entry should be a dictionary with a key for
        each of the parameters contained within the related
        modelspec entry's phi dict. The values should be
        tuples of (str: disribution class, list: distribution xargs).

        Ex: priors = {
                'parameter_a': ('Normal', [0, 1]),
                'parameter_b': ('HalfNormal', [0.5, 0.5])
                }

    For example, given a modelspec with five entries:
        _set_priors(modelspec, [p0, p1, p2])
    Would assign p1, p2, and p3 to modelspec entries 0, 1, and 2, respectively.
    However,
        _set_priors(modelspec, [p0, p1, p2], i=2, j=5)
    Would assign p1, p2, and p3 to entries 2, 3, and 4, respectively.

    If the length of the priors list is greater than the length
    of the modelspec, the excess priors will be ignored with
    a warning.
    """
    if len(priors) > len(modelspec):
        raise RuntimeWarning("More priors than modelspec entries,"
                             "priors list will be truncated.")
        priors = priors[:len(modelspec)]

    for m, p in zip(modelspec[i:j], priors):
        m['priors'] = p

def default_priors(rec, modelspec):
    """
    Given a recording and modelspec, sets the prior distributions for each
    entry to their default specifications
    (defined top-level in nems.priors).
    """
    priors_fns = [m['priors_fn'] for m in modelspec]
    priors_list = [p(rec, m['phi'], m['fn_kwargs'])
                   for p, m in zip(priors_fns, modelspec)]
    set_priors(priors_list)

def phis_from_priors(modelspec, i=0, j=-1):
    """
    Given a modelspec and indices i and j, set the value of each
    phi parameter equal to a sample from that parameter's
    prior distribution.
    """
    for m in modelspec[i:j]:
        for k, v in m['phi']:
            dist_type, dist_args = m['priors'][k]
            dist_class = getattr(dist_api, dist_type)
            dist = dist_class(*dist_args)
            m['phi'][k] = dist.sample()

def sample_phi(modelspec):
    """
    Given a modelspec, return a list of phi dictionaries with
    the value for each parameter of each module sampled
    from their respective distributions.
    """
    raise NotImplementedError

# TODO: Better way to set up these functions? basically just a port
#       of bburan's class-based module code for now.  --jacob  2/11/2018

# NOTE: prior-generating functions below should only require
#       three positional arguments:
#           a recording object
#           a phi dictionary for the corresponding
#           transformation function
#           a dictionary of kwargs for the corresponding
#           transformation function
#       This is to maintain the simplicity of the default_priors function
#       defined in this module.
#
#       Some functions may not actually use one or the other argument,
#       but both should be included in the function signature to
#       ensure compatibility.
def weight_channels_priors(rec, wc_phi, wc_kwargs):
    """
    Given a set of keyword arguments for the weight_channels
    transformation functions, returns a prior distribution specification.

    wc_phi must contain:
        {TODO: may need to change weight channels to take mu and sd
               instead of coefficients?}
    """
    raise NotImplementedError
    # copy pasted below, needs refactor

    '''
    Channel weights are specified in the range [0, 1] with 0 being the
    lowest frequency in the input spectrogram and 1 being the highest.
    Hence, the prior must be constrained to the range [0, 1]. Acceptable
    priors include Uniform and Beta priors.
    In this implementation we use Beta priors such that the center for each
    channel can fall anywhere in the range [0, 1]; however, the first
    channel is most likely to fall at the lower end of the frequency axis
    and the last channel at the higher end of the frequency axis.
    See `plot_probability_distribution` to understand how the weights for
    each work.
    '''
    alpha = []
    beta = []
    i_middle = (self.n_outputs-1)/2
    for i in range(self.n_outputs):
        if i < i_middle:
            # This is one of the low-frequency channels.
            a = self.n_outputs + 1
            b = i + 1
        elif i == i_middle:
            # Center the prior such that the expected value for the channel
            # falls at 0.5
            a = self.n_outputs + 1
            b = self.n_outputs + 1
        else:
            # This is one of the high-frequency channels
            a = self.n_outputs-i
            b = self.n_outputs + 1
        alpha.append(a)
        beta.append(b)

    sd = np.full(self.n_outputs, 0.2)
    return {
        'mu': Beta(alpha, beta),
        'sd': HalfNormal(sd),
    }

def fir_priors(rec, fir_phi, fir_kwargs):
    """
    fir_phi must contain:
        {TODO: same question as for weight channels}
    """
    raise NotImplementedError
    # copy pasted below, needs refactoring

    x = initial_data[self.input_name]
    n_inputs = x.shape[0]
    prior_shape = n_inputs, self.n_taps
    c_mu = np.full(prior_shape, 1/self.n_taps, dtype='float32')
    c_sd = np.ones(prior_shape, dtype='float32')
    return {
        'coefficients': Normal(mu=c_mu, sd=c_sd),
    }

def dexp_priors(rec, dexp_phi, dexp_kwargs):
    """
    dexp_phi must contain these keys:
        base, amplitude, shift, and kappa
    """
    raise NotImplementedError
    # copy pasted below, needs refactoring

    resp = initial_data[self.response_name]
    pred = initial_data[self.input_name]

    base_mu, peak_mu = np.nanpercentile(resp, [2.5, 97.5])
    base_mu = np.clip(base_mu, 0.01, np.inf)

    shift_mu = np.nanmean(pred)
    resp_sd = np.nanstd(resp)
    pred_sd = np.nanstd(pred)

    # In general, kappa needs to be about this value (based on the input) to
    # get a reasonable initial slope. Eventually we can explore a more
    # sensible way to initialize this?
    pred_lb, pred_ub = np.nanpercentile(pred, [2.5, 97.5])
    kappa_sd = 10/(pred_ub-pred_lb)


    return {
        'base': Gamma.from_moments(base_mu, resp_sd*2),
        'amplitude': Gamma.from_moments(peak_mu-base_mu, resp_sd*2),
        'shift': Gamma.from_moments(shift_mu, pred_sd*2),
        'kappa': HalfNormal(kappa_sd),
    }

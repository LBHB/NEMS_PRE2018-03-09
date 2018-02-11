from functools import partial

# default mappings of transformation functions to priors

# TODO: Change these to reasonable values where necessary.
#       I wasn't sure which distributions/means/etc made
#       sense for which parameters.     -jacob  2/10/2018
priors = {
        'nems.modules.weight_channels.weight_channels': {
                'coefficients': ('Normal', [0.0, 1.0])
                },
        'nems.modules.fir.fir_filter': {
                'coefficients': ('Normal', [0.0, 1.0])
                },
        'nems.modules.nonlinearity.double_exponential': {
                'base': ('Normal', [1.0, 1.0]),
                'amplitude': ('Normal', [1.0, 1.0]),
                'shift': ('Normal', [1.0, 1.0]),
                'kappa': ('Normal', [1.0, 1.0])
                }
        }

def _set_priors(modelspec, priors, i=0, j=-1):
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

        Ex: priors_for_wc = {
                    'coefficients': ('Normal', [0, 1])
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

    for m, p in zip(modelspec, priors):
        m['priors'] = p

def _default_priors(modelspec):
    """
    Given a modelspec, sets the prior distributions for each
    entry to their default specifications
    (defined top-level in nems.priors).
    """
    priors_list = [priors[m['fn']] for m in modelspec]
    _set_priors(priors_list)

# TODO: How to properly initialize each module's phi from the priors?
#       Need to dig up previous code for bburan's versions of modules
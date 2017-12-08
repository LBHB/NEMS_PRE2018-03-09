import re
from functools import partial

from .registry import keyword_registry
from nems.utilities.utils import mini_fit

################################################################################
# Stack manipulation
################################################################################
def wc(stack, rank=1, transform=None):
    '''
    Applies a n-channel spectral filter to the data stream.

    Parameters
    ----------
    rank : int
        Number of filters to apply to input stream.
    '''

    module = nm.filters.weight_channels(num_chans=rank, parm_type=transform)
    stack.append(module)


def fir(stack, coefs=10, random=False):
    """
    Adds a temporal bin finite impluse response (FIR) filter to the datastream.
    This filter can serve as either the entire STRF for the cell and be fitted
    as such, or as the temporal filter in the factorized STRF if used in
    conjuction with the weight channel spectral filter.

    Parameters
    ----------
    coefs : int
        Number of coefficients for FIR filter.
    random : bool
        If true, initial FIR coefficients are drawn from a normal distribution
        (mu=0, sigma=0.0025). Otherwise, FIR coefficients are set to 0.
    """
    module = nm.filters.fir(num_coefs=coefs, random=random)
    stack.append(module)
    mini_fit(stack,
             mods=['filters.weight_channels','filters.fir','filters.stp'])


def stp(n_channels=1, u=None, tau=None, normalize=False):
    if normalize:
        module = nm.aux.normalize()
        stack.append(module)
    module = nm.filters.stp(num_channels=n_channels)
    if u is not None:
        u = np.array(u)
        module.u[:] = u
    if tau is not None:
        module.tau[:] = tau


################################################################################
# Keyword registry
################################################################################
def parse_wc(groups):
    if groups[0] == 'g':
        transform = 'gauss'
    elif groups[0] is None:
        transform = None
    else:
        raise ValueError('Unsupported argument')
    rank = int(groups[1])
    return partial(wc, rank=rank, transform=transform)


def parse_fir(groups):
    n_coefs = int(groups[0])
    if groups[1] == 'r':
        random = True
    elif groups[1] is None:
        random = False
    else:
        raise ValueError('Unsupported argument')
    return partial(fir, n_coefs=n_coefs, random=random)


keyword_registry.update({
    # Both the FIR and weight channels keywords are simple enough that they can
    # be composed as a set of arguments.
    re.compile(r'wc(\w)??(\d{2})'): parse_wc,
    re.compile(r'fir(\d{2})(\w)??'): parse_fir,
    'stp1pcon': partial(stp, n_channels=1, u=0.1, tau=0.5),
    'stp2pc': partial(stp, n_channels=2, u=[0.01, 0.1]),
    'stp1pcn': partial(stp, n_channels=1, u=0.01),
})

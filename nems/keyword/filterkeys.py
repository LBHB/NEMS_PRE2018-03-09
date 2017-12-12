import re
from functools import partial

from .registry import keyword_registry
from nems.utilities.utils import mini_fit

from nems.modules import filters

################################################################################
# Stack manipulation
################################################################################
def wc(stack, rank=1):
    '''
    Applies a n-channel spectral filter to the data stream.

    Parameters
    ----------
    rank : int
        Number of filters to apply to input stream.
    '''
    module = filters.WeightChannels(output_channels=rank)
    stack.append(module)


def wc_gaussian(stack, rank=1):
    '''
    Applies a n-channel spectral filter to the data stream.

    Parameters
    ----------
    rank : int
        Number of filters to apply to input stream.
    '''
    module = filters.WeightChannelsGaussian(output_channels=rank)
    stack.append(module)


def fir(stack, n_coefs=10, init_method='zeros'):
    """
    Adds a temporal bin finite impluse response (FIR) filter to the datastream.
    This filter can serve as either the entire STRF for the cell and be fitted
    as such, or as the temporal filter in the factorized STRF if used in
    conjuction with the weight channel spectral filter.
    """
    module = filters.FIR(n_coefficients=n_coefs, init_method=init_method)
    stack.append(module)


def stp(n_channels=1, u=None, tau=None, normalize=False):
    if normalize:
        module = nm.aux.normalize()
        stack.append(module)
    module = filters.stp(num_channels=n_channels)
    if u is not None:
        u = np.array(u)
        module.u[:] = u
    if tau is not None:
        module.tau[:] = tau


################################################################################
# Keyword registry
################################################################################
def parse_wc(groups):
    rank = int(groups[1])
    transform = groups[0]
    if transform is None:
        return partial(wc, rank=rank)
    elif transform == 'g':
        return partial(wc_gaussian, rank=rank)
    else:
        raise ValueError('Unsupported argument')


def parse_fir(groups):
    n_coefs = int(groups[0])
    if groups[1] == 'r':
        init_method = 'random'
    elif groups[1] is None:
        init_method = 'zeros'
    else:
        raise ValueError('Unsupported argument')
    return partial(fir, n_coefs=n_coefs, init_method=init_method)


keyword_registry.update({
    # Both the FIR and weight channels keywords are simple enough that they can
    # be composed as a set of arguments.
    re.compile(r'^wc(\w)??(\d{2})$'): parse_wc,
    re.compile(r'^fir(\d{2})(\w)??$'): parse_fir,
    'stp1pcon': partial(stp, n_channels=1, u=0.1, tau=0.5),
    'stp2pc': partial(stp, n_channels=2, u=[0.01, 0.1]),
    'stp1pcn': partial(stp, n_channels=1, u=0.01),
})

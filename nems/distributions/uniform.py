from scipy import stats

from .distribution import Distribution


class Uniform(Distribution):
    '''
    Uniform prior

    Parameters
    ----------
    lower : scalar or ndarray
        Lower bound of distribution
    upper : scalar or ndarray
        Upper bound of distribution
    '''

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.uniform(self.lower, self.upper)

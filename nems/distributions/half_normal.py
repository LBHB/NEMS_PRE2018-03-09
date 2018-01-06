from scipy import stats

from .distribution import Distribution


class HalfNormal(Distribution):

    def __init__(self, sd, shape=None):
        '''
        Represents a single prior or array of priors

        Parameters
        ----------
        sigma : scalar or ndarray
            Standard deviation of distribution

        Example
        -------
        Define a scalar prior for a single coefficient
        >>> weights = HalfNormal(sd=5)
        >>> weights.mean()
        3

        Define an array of priors with different means but same standard
        deviation
        >>> weights = HalfNormal(sd=[1, 1])
        >>> weights.mean()
        [1, 5]

        Define an array of priors with same mean and standard deviation
        >>> weights = HalfNormal(sd=[1, 1])
        >>> weights.mean()
        [3, 3]
        '''
        self.sd = sd

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.halfnorm(scale=self.sd)

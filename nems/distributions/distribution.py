import numpy as np


class Distribution:
    '''
    Base class for a Distribution
    '''

    @classmethod
    def value_to_string(cls, value):
        if value.ndim == 0:
            return 'scalar'
        else:
            shape = ', '.join(str(v) for v in value.shape)
            return 'array({})'.format(shape)

    def mean(self):
        '''
        Return the expected value of the distribution
        '''
        return self.distribution.mean()

    def sample(self):
        '''
        Return a random sample from the distribution
        '''
        return self.distribution.rvs()

    def percentile(self, percentile):
        '''
        Calculate the percentile

        Parameters
        ----------
        percentile : float [0, 1]
            Probability at which the result is calculated. Should be specified as
            a fraction in the range 0 ... 1 rather than a percent.

        Returns
        -------
        value : float
            Value of random variable at given percentile

        For some distributions (e.g., Normal), the bounds will be +/- infinity.
        In those situations, you can request that you get the bounds for the 99%
        interval to get a slightly more reasonable constraint that can be passed
        to the fitter.

        >>> from nems.distributions.api import Normal
        >>> prior = Normal(mu=0, sd=1)
        >>> lower = prior.percentile(0.005)
        >>> upper = prior.percentile(0.995)
        '''
        return self.distribution.ppf(percentile)

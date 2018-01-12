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

    def get_bounds(self, lower=0, upper=1):
        '''
        Return the bounds of the distribution

        Parameters
        ----------
        lower : float [0, 1]
            Percentile at which the lower bound should be calculated. Should be
            specified as a fraction in the range 0 ... 1 rather than a percent.
        upper : float [0, 1]
            Percentile at which the upper bound should be calculated. Should be
            specified as a fraction in the range 0 ... 1 rather than a percent.

        Returns
        -------
        lower : float
            Lower bound of the distribution.
        upper : float
            Upper bound of the distribution

        For some distributions (e.g., Normal), the bounds will be +/- infinity.
        In those situations, you can request that you get the bounds for the 99%
        interval to get a slightly more reasonable constraint that can be passed
        to the fitter.

        >>> from nems.distributions.api import Normal
        >>> prior = Normal(mu=0, sd=1)
        >>> prior.get_bounds(0.005, 0.995)
        '''
        return self.distribution.ppf([lower, upper])

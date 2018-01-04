class Normal(Distribution):

    def __init__(self, mu, sigma, shape=None):
        '''
        Represents a single prior or array of priors

        Parameters
        ----------
        mu : scalar or ndarray
            Mean of Normal distribution
        sigma : scalar or ndarray
            Standard deviation of distribution
        shape : None, integer or tuple of integers
            If None, this prior represents a scalar value. If integer, this is a
            1D array of priors. If a tuple, this is a n-dimensional array of
            priors.

        Example
        -------
        Define a scalar prior for a single coefficient
        >>> weights = Normal(mu=3, sd=5)
        >>> weights.mean()
        3

        Define an array of priors with different means but same standard
        deviation
        >>> weights = Normal(mu=[1, 5], sd=1, shape=2)
        >>> weights.mean()
        [1, 5]

        Define an array of priors with same mean and standard deviation
        >>> weights = Normal(mu=3, sd=1, shape=2)
        >>> weights.mean()
        [3, 3]
        '''
        self.mu = expand_scalar(mu, shape)
        self.sigma = expand_scalar(sigma, shape)
        self.shape = shape

        # The beauty of using the scipy stats module is it correctly handles
        # scalar and multidimensional arrays of distributions.
        self.distribution = stats.normal(mu=self.mu, sigma=self.sigma)

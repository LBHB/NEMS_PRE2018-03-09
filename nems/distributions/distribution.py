import numpy as np
import matplotlib.pyplot as plt


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

    @property
    def shape(self):
        return self.mean().shape

    def sample(self, size=1):
        n = self.shape()
        return self.distribution.rvs(size=(size, n[0]))

    def pdf(self, x):
        return self.distribution.pdf(x)

    def plot(self):
        x_min = self.percentile(0.01)
        x_max = self.percentile(0.99)
        n = self.shape[0]

        xs, _ = np.mgrid[xmin:xmax:100j, 1:n+1]
        ys = self.pdf(xs)

        labels = ["phi[{}]".format(i) for i in range(n+1)]
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, ys, alpha=0.7, lw=2)
        ax.legend(loc='best', frameon=False, labels=labels)
        plt.show()

import numpy as np
import matplotlib.pyplot as plt

class Distribution:

    def mean(self):
        return self.distribution.mean()

    def len(self):
        return len(self.shape())

    def shape(self):
        return self.mean().shape

    def sample(self, size=1):
        n = self.shape()
        return self.distribution.rvs(size=(size, n[0]))

    def pdf(self, x):
        return self.distribution.pdf(x)

    def ppf(self, frac):
        return self.distribution.ppf(frac)

    def plot(self):
        xmin = min(self.ppf(0.01))
        xmax = max(self.ppf(0.99))
        n = self.len()
        xs, _ = np.mgrid[xmin:xmax:100j, 1:n+1]
        ys = self.pdf(xs)
        labels = ["phi[{}]".format(i) for i in range(n+1)]
        fig, ax = plt.subplots(1, 1)
        ax.plot(xs, ys, alpha=0.7, lw=2)
        ax.legend(loc='best', frameon=False, labels=labels)
        plt.show()


import numpy as np
import pylab as pl

from ..distributions.api import Gamma, HalfNormal
from .module import Module


class Nonlinearity(Module):
    pass


def double_exponential(x, base, amplitude, shift, kappa):
    '''
    Double exponential function
    '''
    return base + amplitude * np.exp(-np.exp(-kappa * (x-shift)))


class DoubleExponential(Nonlinearity):

    def __init__(self, input_name='pred', output_name='pred',
                 response_name='pred'):
        self.input_name = input_name
        self.output_name = output_name
        self.response_name = response_name

    def get_priors(self, initial_data):
        resp = initial_data[self.response_name]
        pred = initial_data[self.input_name]

        base_mu, peak_mu = np.nanpercentile(resp, [2.5, 97.5])
        base_mu = np.clip(base_mu, 0.01, np.inf)

        shift_mu = np.nanmean(pred)
        resp_sd = np.nanstd(resp)
        pred_sd = np.nanstd(pred)

        # In general, kappa needs to be about this value (based on the input) to
        # get a reasonable initial slope. Eventually we can explore a more
        # sensible way to initialize this?
        pred_lb, pred_ub = np.nanpercentile(pred, [2.5, 97.5])
        kappa_sd = 10/(pred_ub-pred_lb)


        return {
            'base': Gamma.from_moments(base_mu, resp_sd*2),
            'amplitude': Gamma.from_moments(peak_mu-base_mu, resp_sd*2),
            'shift': Gamma.from_moments(shift_mu, pred_sd*2),
            'kappa': HalfNormal(kappa_sd),
        }

    def evaluate(self, data, phi):
        x = data[self.input_name]
        return {
            self.output_name: double_exponential(x, **phi)
        }

    def plot_coefficients(self, phi, data=None, axes=None):
        if data is not None:
            pred = data[self.input_name]
            x = np.linspace(pred.min(), pred.max(), 100)
        else:
            x = np.linspace(0, 1000, 100)

        if axes is None:
            ax = pl.gca()

        y = double_exponential(x, **phi)
        ax.plot(x, y, 'k-')

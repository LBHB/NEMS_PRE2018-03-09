import numpy as np

from ..distributions.api import Normal, HalfNormal
from .module import Module


class Nonlinearity(Module):
    pass


def double_exponential(x, base, amplitude, shift, kappa):
    '''
    Double exponential function
    '''
    return base + amplitude * np.exp(-np.exp(-kappa * (x-shift)))


class DoubleExponential(Nonlinearity):

    def __init__(self, input_name, output_name, response_name):
        self.input_name = input_name
        self.output_name = output_name
        self.response_name = response_name

    def get_inputs(self):
        return {
            self.input_name: (Ellipsis, -1, -1, -1),
        }

    def get_outputs(self):
        return {
            self.output_name: (Ellipsis, -1, -1, -1),
        }

    def get_priors(self, initial_data):
        resp = initial_data[self.response_name]
        pred = initial_data[self.input_name]
        base_mu, peak_mu = np.nanpercentile(resp, [2.5, 97.5])
        shift_mu = np.nanmean(pred)
        resp_sd = np.nanstd(resp)
        pred_sd = np.nanstd(pred)

        # In general, kappa needs to be about this value (based on the input) to
        # get a reasonable initial slope. Eventually we can explore a more
        # sensible way to initialize this?
        pred_lb, pred_ub = np.nanpercentile(pred, [2.5, 97.5])
        kappa_sd = 10/(pred_ub-pred_lb)

        return {
            'base': Normal(mu=base_mu, sd=resp_sd),
            'amplitude': Normal(mu=peak_mu-base_mu, sd=resp_sd),
            'shift': Normal(mu=shift_mu, sd=pred_sd),
            'kappa': HalfNormal(sd=kappa_sd),
        }

    def evaluate(self, data, phi):
        x = data[self.input_name]
        return {
            self.output_name: double_exponential(x, **phi)
        }

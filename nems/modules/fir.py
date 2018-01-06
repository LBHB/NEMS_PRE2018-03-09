import numpy as np
from scipy import signal

from ..distributions.api import Normal
from .module import Module


def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    null_data = np.full(n_taps*2, x[0])
    zi = np.ones(n_taps-1)
    return signal.lfilter(b, [1], null_data, zi=zi)[1]


def fir_filter(x, coefficients):
    result = []
    for x, c in zip(x, coefficients):
        zi = get_zi(c, x)
        r, zf = signal.lfilter(c, [1], x, zi=zi)
        result.append(r[np.newaxis])
    result = np.concatenate(result)
    return np.sum(result, axis=-2, keepdims=True)


class FIR(Module):

    def __init__(self, n_taps, input_name, output_name):
        self.n_taps = n_taps
        self.input_name = input_name
        self.output_name = output_name

    def get_inputs(self):
        return {
            self.input_name: (Ellipsis, -1, -1, -1),
        }

    def get_outputs(self):
        return {
            self.output_name: (Ellipsis, -1, -1, -1),
        }

    def evaluate(self, data, phi):
        coefficients = phi['coefficients']
        x = data[self.input_name]
        return {
            self.output_name: fir_filter(x, coefficients)
        }

    def get_priors(self, initial_data):
        x = initial_data[self.input_name]
        n_inputs = x.shape[0]
        prior_shape = n_inputs, self.n_taps
        c_mu = np.full(prior_shape, 1/self.n_taps)
        c_sd = np.ones(prior_shape)
        return {
            'coefficients': Normal(mu=c_mu, sd=c_sd),
        }

import numpy as np
from scipy import stats

from ..distributions.api import Normal, HalfNormal
from .module import Module


def weight_channels(x, weights):
    '''
    Parameters
    ----------
    x : ndarray
        The last two axes must map to channel x time. Any remaning dimensions
        will be passed through. Weighting will be applied to the channel
        dimension.
    coefficients : 2d array (output channel x input channel weights)
        Weighting of the input channels. A set of weights are provided for each
        desired output channel. Each row in the array are the weights for the
        input channels for that given output. The length of the row must be
        equal to the number of channels in the input array
        (e.g., `x.shape[-3] == coefficients.shape[-1]`).
    baseline : 1d array
        Offset of the channels

    Returns
    -------
    out : ndarray
        Result of the weight channels transform. The shape of the output array
        will be equal to the input array except for the third to last dimension
        (the channel dimension). This dimension's length will be equivalent to
        the length of the coefficients.
    '''
    return weights @ x


class BaseWeightChannels(Module):

    def __init__(self, n_outputs, input_name, output_name):
        self.n_outputs = n_outputs
        self.input_name = input_name
        self.output_name = output_name

    def get_inputs(self):
        return {
            self.input_name: (Ellipsis, -1, -1, -1),
        }

    def get_outputs(self):
        return {
            self.output_name: (Ellipsis, self.n_outputs, -1, -1),
        }

    def evaluate(self, data, phi):
        x = data[self.input_name]
        weights = self.get_weights(x.shape[0], phi)
        return {
            self.output_name: weight_channels(x, weights)
        }

    def from_json(self, json_dict):
        self.n_channels = json_dict['n_channels']


class WeightChannels(BaseWeightChannels):
    '''
    Nonparameterized channel weights
    '''
    pass


class WeightChannelsGaussian(BaseWeightChannels):
    '''
    Parameterized channel weights
    '''

    def get_priors(self, initial_data):
        # Space the priors for each output channel evenly along the normalized
        # frequency axis (where 0 is the min frequency and 1 is the max
        # frequency).
        mu = np.linspace(0, 1, self.n_outputs)
        mu_sd = np.full_like(mu, 0.5)
        sd = np.full_like(mu, 0.2)
        return {
            'mu': Normal(mu=mu, sd=mu_sd),
            'sd': HalfNormal(sd=sd),
        }

    def get_weights(self, n, phi):
        mu = phi['mu']
        sd = phi['sd']

        # Add a half step to the array so that x represents the bin "centers".
        x = np.arange(n, dtype=np.float)/n + 0.5/n
        weights = [stats.norm.pdf(x, loc=m, scale=s) for m, s in zip(mu, sd)]

        # Normalize so the sum of the weights equals 1
        weights = np.array(weights)/n
        return weights

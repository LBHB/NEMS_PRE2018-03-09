import numpy as np
from scipy import stats

from theano import tensor

from ..distributions.api import Beta, HalfNormal
from .module import Module


def plot_probability_distribution(n_outputs):
    '''
    Illustrates how the Beta priors for the gaussian weight channel centers work
    by plotting the probability distribution for each.
    '''
    x = np.arange(0, 1, 1000)

    middle_index = (n_outputs-1)/2
    for i in range(n_outputs):
        alpha = n_outputs + 1
        beta = i + 1
        print(i, (n_outputs-1)/2)
        if i < (n_outputs-1)/2:
            alpha = n_outputs + 1
            beta = i + 1
        elif i == (n_outputs-1)/2:
            alpha = n_outputs + 1
            beta = n_outputs + 1
        else:
            beta = n_outputs + 1
            alpha = n_outputs-i

        y = stats.beta(alpha, beta).pdf(x)
        pl.plot(x, y, label='Channel ' + str(i))

    pl.legend()


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

    def evaluate(self, data, phi):
        x = data[self.input_name]
        n = x.shape[0]
        # Add a half step to the array so that x represents the bin "centers".
        channel_centers = np.arange(n)/n + 0.5/n
        weights = self.get_weights(channel_centers, phi)
        return {
            self.output_name: weight_channels(x, weights)
        }

    def generate_tensor(self, data, phi):
        x = data[self.input_name]
        n = x.shape[0]
        # Add a half step to the array so that x represents the bin "centers".
        #x = np.arange(n)/n + 0.5/n
        channel_centers = tensor.arange(n)/n + 0.5/n
        weights = self.get_weights(channel_centers, phi)
        return {
            self.output_name: tensor.dot(weights, x)
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
        '''
        Channel weights are specified in the range [0, 1] with 0 being the
        lowest frequency in the input spectrogram and 1 being the highest.
        Hence, the prior must be constrained to the range [0, 1]. Acceptable
        priors include Uniform and Beta priors.

        In this implementation we use Beta priors such that the center for each
        channel can fall anywhere in the range [0, 1]; however, the first
        channel is most likely to fall at the lower end of the frequency axis
        and the last channel at the higher end of the frequency axis.

        See `plot_probability_distribution` to understand how the weights for
        each work.
        '''
        alpha = []
        beta = []
        i_middle = (self.n_outputs-1)/2
        for i in range(self.n_outputs):
            if i < i_middle:
                # This is one of the low-frequency channels.
                a = self.n_outputs + 1
                b = i + 1
            elif i == i_middle:
                # Center the prior such that the expected value for the channel
                # falls at 0.5
                a = self.n_outputs + 1
                b = self.n_outputs + 1
            else:
                # This is one of the high-frequency channels
                a = self.n_outputs-i
                b = self.n_outputs + 1
            alpha.append(a)
            beta.append(b)

        sd = np.full(self.n_outputs, 0.2)
        return {
            'mu': Beta(alpha, beta),
            'sd': HalfNormal(sd),
        }

    def get_weights(self, x, phi):

        # Set up so that the channels are broadcast along the last dimension
        # (columns) and each row represents the weights for one of the output
        # channels.
        mu = phi['mu'][..., np.newaxis]
        sd = phi['sd'][..., np.newaxis]
        weights = 1/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mu)/sd)**2)
        return weights.astype('float32')


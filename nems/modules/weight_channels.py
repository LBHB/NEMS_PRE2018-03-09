from .module import Module


def weight_channels(x, weights):
    '''
    Parameters
    ----------
    x : ndarray
        The last three axes must map to channel x trial x time. Any remaning
        dimensions will be passed through. Weighting will be applied to the
        channel dimension.
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
    # We need to shift the channel dimension to the second-to-last dimension
    # so that matmul will work properly (it operates on the last two
    # dimensions of the inputs and treats the rest of the dimensions as
    # stacked matrices).
    x = np.swapaxes(x, -3, -2)
    x = weights @ x
    x = np.swapaxes(x, -3, -2)
    return x


class BaseWeightChannels(Module):

    def __init__(self, n_outputs):
        self.n_outputs = n_outputs

    def get_inputs(self):
        return {
            'pred': (Ellipsis, -1, -1, -1),
        }

    def get_outputs(self):
        return {
            'pred': (Ellipsis, self.n_outputs, -1, -1),
        }

    def evaluate(self, data, phi):
        weights = self.get_weights(phi)
        return {
            'pred': weight_channels(data['pred'], weights)
        }

    def from_json(self, json_dict):
        self.n_channels = json_dict['n_channels']


class NPWeightChannels(BaseWeightChannels):
    '''
    Nonparameterized channel weights
    '''
    pass


class GaussianWeightChannels(BaseWeightChannels):
    '''
    Parameterized channel weights
    '''

    def get_priors(self, initial_data):
        # Space the priors for each output channel evenly along the normalized
        # frequency axis (where 0 is the min frequency and 1 is the max
        # frequency).
        mu = np.linspace(0, 1, self.output_channels)
        mu_sd = np.full_like(mu, 0.5)
        sd = np.full_like(mu, 0.5)
        return {
            'mu': Normal(mu=mu, sd=mu_sd),
            'sigma': HalfNormal(sd=sd),
        }

    def get_weights(self, phi):
        raise NotImplementedError

import nems.Module

###############################################################################
# Module structure
###############################################################################
def weight_channels(x, mu, sigma):
    # The actual computation is in a stand-alone function that can be readily
    # re-used by anyone who wants to perform the operation without having to
    # wrap the data into a Signal object (also facilitates unit-tests).
    return np.mean((x-mu)/sigma, axis=-3, keepdims=True)


class WeightChannelsGaussian(Module):
    # Ellipsis means that any number of dimensions are allowed, -1 means that
    # the actual shape of that dimension doesn't matter. Here, this means that
    # we can "stack" multiple neurons into a 4D array (neuron, frequency, trial,
    # time), thereby making it possible to fit population models without silly
    # `for x in self.d_in` stuff.

    def __init__(self, output_channels):
        # Number of output channels
        self.output_channels = output_channels

    def evaluate(self, data, phi):
        # Function that pulls information out of the Signals dictionary and phi,
        # performs the computation and stores it back in the Signals dictionary.
        x = data['stim']
        state = data['state']
        mu = phi['mu']
        sigma = phi['sigma']
        stim = weight_channels(x, mu, sigma)
        return {'stim': stim}

    def get_priors(self, initial_data):
        # Initial data is provided in case this helps properly initialize the
        # prior (e.g., this can be helpful when constructing the prior for the
        # nonlinear functions that need to know the min/max of the observed
        # firing rate).
        mu = []
        sd = []
        for i in range(self.output_channels):
            mu_prior = Normal(mu=(i+1)/(self.output_channels+1), sd=0.5)
            mu.append(mu_prior)
            sd_prior = HalfNormal(sd=0.5)
            sd.append(sd_prior)
        return {
            'mu': mu,
            'sd': sd,
        }

    @property
    def output_signals(self):
        return {
            'stim': (Ellipsis, self.output_channels, -1, -1)
        }

    @property
    def input_signals(self):
        return {
            'stim': (Ellipsis, -1, -1, -1),
        }

    def to_json(self):
        return {
            'output_channels': self.output_channels
            'output_signals': self.output_signals,
            'input_signals': self.input_signals,
        }

    def from_json(self, json_dict):
        # Skip the output_signals and input_signals because these are
        # automatically calculated using the output_channels attribute. We
        # include them in the to_json so that the JSON file contains extra
        # information about the model that can help others understand what's
        # going on if they were to read the JSON file.
        self.output_channels = json_dict['output_channels']

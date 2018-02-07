from nems.utils import split_keywords
from nems import keywords


def from_keywords(keyword_string, registry=keywords.defaults):
    '''
    Returns a modelspec created by splitting keyword_string on underscores
    and replacing each keyword with what is found in the nems.keywords.defaults
    registry. You may provide your own keyword registry using the
    registry={...} argument.
    '''
    keywords = split_keywords(keyword_string)

    # Lookup the modelspec fragments in the registry
    modelspec = []
    for kw in keywords:
        d = registry[kw]
        d['id'] = kw
        modelspec.append(d)

    return modelspec



################################################################################
### DEXP 
# from ..distributions.api import Gamma, HalfNormal
# def initialize_dexp(initial_data, i=None, o=None, **kwargs):
#         resp = initial_data[i]

#         base_mu, peak_mu = np.nanpercentile(resp, [2.5, 97.5])
#         base_mu = np.clip(base_mu, 0.01, np.inf)

#         shift_mu = np.nanmean(pred)
#         resp_sd = np.nanstd(resp)
#         pred_sd = np.nanstd(pred)

#         # In general, kappa needs to be about this value (based on the input) to
#         # get a reasonable initial slope. Eventually we can explore a more
#         # sensible way to initialize this?
#         pred_lb, pred_ub = np.nanpercentile(pred, [2.5, 97.5])
#         kappa_sd = 10/(pred_ub-pred_lb)


#         return {
#             'base': Gamma.from_moments(base_mu, resp_sd*2),
#             'amplitude': Gamma.from_moments(peak_mu-base_mu, resp_sd*2),
#             'shift': Gamma.from_moments(shift_mu, pred_sd*2),
#             'kappa': HalfNormal(kappa_sd),
#         }

################################################################################
## WEIGHT CHANNELS

# class WeightChannelsGaussian(BaseWeightChannels):
#     '''
#     Parameterized channel weights
#     '''

#     def get_priors(self, initial_data):
#         '''
#         Channel weights are specified in the range [0, 1] with 0 being the
#         lowest frequency in the input spectrogram and 1 being the highest.
#         Hence, the prior must be constrained to the range [0, 1]. Acceptable
#         priors include Uniform and Beta priors.

#         In this implementation we use Beta priors such that the center for each
#         channel can fall anywhere in the range [0, 1]; however, the first
#         channel is most likely to fall at the lower end of the frequency axis
#         and the last channel at the higher end of the frequency axis.

#         See `plot_probability_distribution` to understand how the weights for
#         each work.
#         '''
#         alpha = []
#         beta = []
#         i_middle = (self.n_outputs-1)/2
#         for i in range(self.n_outputs):
#             if i < i_middle:
#                 # This is one of the low-frequency channels.
#                 a = self.n_outputs + 1
#                 b = i + 1
#             elif i == i_middle:
#                 # Center the prior such that the expected value for the channel
#                 # falls at 0.5
#                 a = self.n_outputs + 1
#                 b = self.n_outputs + 1
#             else:
#                 # This is one of the high-frequency channels
#                 a = self.n_outputs-i
#                 b = self.n_outputs + 1
#             alpha.append(a)
#             beta.append(b)

#         sd = np.full(self.n_outputs, 0.2)
#         return {
#             'mu': Beta(alpha, beta),
#             'sd': HalfNormal(sd),
#         }
    # def get_weights(self, x, phi):
    #     # Set up so that the channels are broadcast along the last dimension
    #     # (columns) and each row represents the weights for one of the output
    #     # channels.
    #     mu = phi['mu'][..., np.newaxis]
    #     sd = phi['sd'][..., np.newaxis]
    #     weights = 1/(sd*np.sqrt(2*np.pi)) * np.exp(-0.5 * ((x-mu)/sd)**2)
    #     return weights.astype('float32')


################################################################################@
## FIR FILTER
    # def get_priors(self, initial_data):
    #     x = initial_data[self.input_name]
    #     n_inputs = x.shape[0]
    #     prior_shape = n_inputs, self.n_taps
    #     c_mu = np.full(prior_shape, 1/self.n_taps, dtype='float32')
    #     c_sd = np.ones(prior_shape, dtype='float32')
    #     return {
    #         'coefficients': Normal(mu=c_mu, sd=c_sd),
    #     }

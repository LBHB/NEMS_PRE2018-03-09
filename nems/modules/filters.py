import numpy as np
from scipy import signal
from scipy.stats import norm


from nems.modules.base import Module
import nems.utilities.utils


################################################################################
# Base class for all filters in this module
################################################################################
class Filter(Module):

    plot_fns = Module.plot_fns + [nems.utilities.plot.plot_strf]


################################################################################
# Channel weighting
################################################################################
def weight_channels(x, weights, y_offset=None):
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
    y_offset : 1d array
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
    if y_offset is not None:
        x += y_offset[..., np.newaxis, np.newaxis]
    return x


class WeightChannels(Module):
    """
    Apply a weighting matrix across a variable in the data stream.
    """

    name = 'filters.weight_channels'
    fit_fields = ['phi', 'y_offset']
    user_editable_fields = Module.user_editable_fields + fit_fields
    plot_fns = Module.plot_fns + [nems.utilities.plot.plot_strf]

    output_channels = None
    y_offset =  None
    phi = None

    def my_init(self, output_channels, *args, **kwargs):
        self.output_channels = output_channels

    def initialize_parameters(self):
        input_data = self.get_input()
        input_channels = input_data.shape[-3]
        self.y_offset = np.zeros(n_channels)
        phi_shape = [self.output_channels, input_channels]
        self.phi = np.random.normal(1, 0.1, phi_shape)

    def get_weights(self):
        return self.phi

    def get_y_offset(self):
        return self.y_offset

    def my_eval(self, x):
        weights = self.get_weights()
        y_offset = self.get_y_offset()
        return weight_channels(x, weights, y_offset)


class WeightChannelsGaussian(WeightChannels):

    name = 'filters.weight_channels_gaussian'
    input_channels = None

    def initialize_parameters(self):
        input_data = self.get_input()
        input_channels = input_data.shape[-3]
        self.y_offset = np.zeros(input_channels)
        mu = np.random.uniform(high=input_channels, size=self.output_channels)
        sigma = np.full_like(mu, fill_value=input_channels/4.0)
        self.phi = np.c_[mu, sigma]
        self.input_channels = input_channels

    def get_weights(self):
        # Pull values for mu and sigma out of phi. The length of mu and sigma
        # are equal to the number of output channels.
        mu_khz = self.phi[0]
        sigma_khz = self.phi[1]

        # Check if mu falls outside 0.2 to 20 kHz
        if np.any(mu_khz < 0.2) or np.any(mu_khz > 20):
            raise ValueError('Invalid coefficient')

        spacing = np.log10(20e3) - np.log10(200) / self.input_channels
        mu = (np.log10(mu_khz*1e3) - np.log10(200)) / spacing
        sigma = (sigma_khz/10)*mu

        # a1*exp(-((x-b1)/c1)^2)

        x = np.arange(self.input_channels)
        coefs = norm.pdf(x, mu[..., np.newaxis], sigma[..., np.newaxis])
        mu = mu[..., np.newaxis]
        sigma = sigma[..., np.newaxis]
        coefs = np.exp(-np.square((x-mu)/sigma))
        coefs /= coefs.sum(axis=-1)[..., np.newaxis]
        print(coefs.sum(axis=-1))
        print(coefs.shape)
        return coefs


################################################################################
# FIR filter
################################################################################
def get_zi(b, x):
    # This is the approach NARF uses. If the initial value of x[0] is 1,
    # this is identical to the NEMS approach. We need to provide zi to
    # lfilter to force it to return the final coefficients of the dummy
    # filter operation.
    n_taps = len(b)
    null_data = np.full(n_taps*2, x[0])
    zi = np.ones(n_taps-1)
    return signal.lfilter(b, [1], null_data, zi=zi)[1]


def fir_filter(x, coefficients, baseline=None):
    # TODO: This will be a nice addition, but for now let's leave it out
    # because NARF doesn't do this. Alternatively, Ivar's revamp of the data
    # loading system may  make this a moot point.
    #pad_width = coefs.shape[-1] * 2
    #padding = [(0, 0)] * X.ndim
    #padding[-1] = (0, pad_width)
    #X = np.pad(X, padding, mode='constant')

    result = []
    x = x.swapaxes(0, -3)
    for x, c in zip(x, coefficients):
        old_shape = x.shape
        x = x.ravel()
        zi = get_zi(c, x)
        r, zf = signal.lfilter(c, [1], x, zi=zi)
        r.shape = old_shape
        result.append(r[np.newaxis])
    result = np.concatenate(result)
    result = np.sum(result, axis=0)
    if baseline is not None:
        result += baseline
    return result


class FIR(Module):
    """
    Applies a linear FIR filter
    """
    name = 'filters.fir'
    user_editable_fields = Module.user_editable_fields + ['n_coefs']
    fit_fields = ['baseline', 'coefficients']

    coefficients = None
    baseline = None

    def initialize_coefficients(self):
        input_data = self.get_input()
        input_channels = input_data.shape[-3]
        coef_shape = [input_channels, self.n_coefficients]
        if init_method == 'zeros':
            coefs = np.zeros(coef_shape)
        elif init_method == 'random':
            coefs = np.random.normal(0, 2.5e-3, size=coef_shape)
        self.baseline = 0

    def my_init(self, n_coefficients=20, init_method='zeros'):
        self.n_coefficients = n_coefficients
        self.init_method = init_method

    def get_coefficients(self):
        return self.coefficients

    def get_baseline(self):
        return self.baseline

    def my_eval(self, x):
        coefficients = self.get_coefficients()
        baseline = self.get_baseline()
        return fir_filter(x, coefficients, baseline)


################################################################################
# STP filter
################################################################################
# TODO: update the STP code!
class stp(Module):
    """
    stp - simulate short-term plasticity with the Tsodyks and Markram model

    m.editable_fields = {'num_channels', 'strength', 'tau', 'strength2', 'tau2',...
                    'per_channel', 'offset_in', 'facil_on', 'crosstalk',...
                    'input', 'input_mod','time', 'output' };
    """
    name='filters.stp'
    user_editable_fields=['input_name','output_name','fit_fields','num_channels','u','tau','offset_in','deponly','crosstalk']
    plot_fns=[nems.utilities.plot.pre_post_psth, nems.utilities.plot.plot_spectrogram]
    coefs=None
    baseline=0
    u=np.zeros([1,1])
    tau=np.zeros([1,1])+0.1
    offset_in=np.zeros([1,1])
    crosstalk=0
    dep_only=False
    num_channels=1
    num_dims=1

    def my_init(self, num_dims=0, num_channels=1, u=None, tau=None, offset_in=None,
                crosstalk=0, fit_fields=['tau','u']):
        """
        num_channels:
        u:
        tau:
        """
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        if self.d_in and not(num_dims):
            num_dims=self.d_in[0][self.input_name].shape[0]
        Zmat=np.zeros([num_dims,num_channels])
        if not u:
            u=Zmat
        if not tau:
            tau=Zmat+0.1
        if not offset_in:
            offset_in=Zmat

        self.num_dims=num_dims
        self.num_channels=num_channels
        self.fit_fields=fit_fields
        self.do_trial_plot=self.plot_fns[0]

        # stp parameters should be matrices num_dims X num_channels or 1 X num_channels,
        # and in the latter case be replicated across num_dims
        self.u=u
        self.tau=tau
        self.offset_in=offset_in
        self.crosstalk=crosstalk

    def my_eval(self,X):
        s=X.shape

        tstim=(X>0)*X;

        # TODO : enable crosstalk

        # TODO : for each stp channel, current just forcing 1
        Y=np.zeros([0,s[1],s[2]])
        di=np.ones(s)
        for j in range(0,self.num_channels):
            ui=np.absolute(self.u[:,j])  # force only depression, no facilitation
            #ui=self.u[:,j]

            # convert tau units from sec to bins
            taui=np.absolute(self.tau[:,j])*self.d_in[0]['fs']


            # go through each stimulus channel
            for i in range(0,s[0]):

                # limits, assumes input (X) range is approximately -1 to +1
                if ui[i]>0.5:
                    ui[i]=0.5
                elif ui[i]<-0.5:
                    ui[i]=-0.5
                if taui[i]<0.5:
                    taui[i]=0.5

                for tt in range(1,s[2]):
                    td=di[i,:,tt-1]  # previous time bin depression
                    if ui[i]>0:
                        delta=(1-td)/taui[i] - ui[i]*td*tstim[i,:,tt-1]
                        td=td+delta
                        td[td<0]=0
                    else:
                        delta=(1-td)/taui[i] - ui[i]*td*tstim[i,:,tt-1]
                        td=td+delta
                        td[td<1]=1
                    di[i,:,tt]=td

            Y=np.append(Y,di*X,0)
            #print(np.sum(np.isnan(Y),1))
            #print(np.sum(np.isnan(di*X),1))

        #plt.figure()
        #pre, =plt.plot(X[0,0,:],label='Pre-nonlinearity')
        #post, =plt.plot(Y[0,0,:],'r',label='Post-nonlinearity')
        #plt.legend(handles=[pre,post])

        return Y



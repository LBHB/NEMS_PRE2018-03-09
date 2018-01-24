from nems.data.api import signal_like, recording_like

from nems.modules.base import nems_module as Module
import nems.utilities.utils

import numpy as np
from scipy import signal


################################################################################
# Channel weighting
################################################################################
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

    Returns
    -------
    out : ndarray
        Result of the weight channels transform. The shape of the output array
        will be equal to the input array except for the third to last dimension
        (the channel dimension). This dimension's length will be equivalent to
        the length of the coefficients.
    '''
    return weights @ x


class WeightChannels(Module):
    """
    weight_channels - apply a weighting matrix across a variable in the data
    stream. Used to provide spectral filters, directly imported from NARF.
    a helper function parm_fun can be defined to parameterize the weighting
    matrix. but by default the weights are each independent
    """
    name = 'filters.weight_channels'
    user_editable_fields = ['input_name', 'output_name', 'fit_fields',
                            'num_dims', 'num_chans', 'coefs', 'phi', 'parm_fun']
    plot_fns = [nems.utilities.plot.plot_strf,
                nems.utilities.plot.plot_spectrogram]

    parm_fun = None
    parm_type = None
    output_channels = 1

    input_name = 'pred'
    output_name = 'pred'

    def init(self, recording):
        super().init(recording)
        if self.parm_type == 'gauss':
            self.parm_fun = self.gauss_fn
            m = np.linspace(1, self.output_channels, self.output_channels + 2)
            m = m[..., np.newaxis]
            m = m[:, 1:-1] / self.input_channels
            s = np.ones([self.output_channels, 1]) / 4
            phi = np.concatenate([m.transpose(), s], 1)
            self.phi = phi
            self.coefs = self.parm_fun(phi)
            self.fit_fields = ['phi']
        else:
            shape = (self.output_channels, self.input_channels)
            self.coefs = np.random.normal(1, 0.1, shape) / self.input_channels
            self.fit_fields = ['coefs']

    def gauss_fn(self, phi):
        coefs = np.zeros([self.num_chans, self.num_dims])
        for i in range(0, self.num_chans):
            m = phi[i, 0] * self.num_dims
            s = phi[i, 1] * self.num_dims
            if s < 0.05:
                s = 0.05
            if (m < 0 and m < s):
                s = -m
            elif (m > self.num_dims and m > self.num_dims + s):
                s = m - self.num_dims

            x = np.arange(0, self.num_dims)
            coefs[i, :] = np.exp(-np.square((x - m) / s))
            coefs[i, :] = coefs[i, :] / np.sum(coefs[i, :])
        return coefs

    def get_coefs(self):
        if self.parm_fun:
            self.coefs = self.parm_fun(self.phi)
            coefs = self.coefs
        else:
            coefs = self.coefs
        return self.coefs

    def evaluate(self, recording, mode):
        # Extract the Numpy array from the signal we need in the recording.
        x_signal = recording.get_signal(self.input_name)
        x = x_signal.as_continuous()
        coefs = self.get_coefs()

        # Do the computation. This returns a Numpy array.
        y = weight_channels(x, coefs)

        # Check if we need to normalize the data.
        if self.norm_output:
            y = self.normalize(x, mode)

        # Now pack this array into a signal object that looks like the original
        # signal, but with updated channel names.
        chans = [str(i) for i in range(coefs.shape[-1])]
        y_signal = signal_like(x_signal, y, chans=chans)

        # Return a copy of the recording containing the new signal.
        return recording_like(recording, {self.output_name: y_signal})


################################################################################
# FIR filtering
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


def fir_filter(x, coefficients, baseline=None, bank_count=1):

    result = []
    for x, c in zip(x, coefficients):
        old_shape = x.shape
        x = x.ravel()
        zi = get_zi(c, x)
        r, zf = signal.lfilter(c, [1], x, zi=zi)
        r.shape = old_shape
        result.append(r[np.newaxis])
    result = np.concatenate(result)

    if bank_count>1:
        # Do not remove the NotImplementedError until this is tested properly
        raise NotImplementedError
        # reshape inputs so that filter is summed separately across each bank
        # need to test this!
        s=list(result.shape)
        #print(s)
        ts0=np.int(s[-3]/bank_count)
        ts1=bank_count
        #print("{0},{1}".format(ts0,ts1))
        result=np.reshape(result,s[:-4]+[ts0,ts1]+s[-2:])
        result = np.sum(result, axis=-4)

    result = np.sum(result, axis=0, keepdims=True)

    if baseline is not None:
        result += baseline

    return result


class FIR(Module):
    """
    fir - the workhorse linear fir filter module. Takes in a 3D stim array
    (channels,stims,time), convolves with FIR coefficients, applies a baseline DC
    offset, and outputs a 2D stim array (stims,time).
    """

    name = 'filters.fir'
    user_editable_fields = ['input_name', 'output_name', 'fit_fields',
                            'num_dims', 'num_coefs', 'coefs', 'baseline',
                            'random_init', 'bank_count']
    plot_fns = [nems.utilities.plot.plot_strf,
                nems.utilities.plot.plot_spectrogram]

    coefs = None
    num_coefs = 20
    baseline = 0
    fit_fields = ['baseline', 'coefs']
    random_init = False
    bank_count = 1

    input_name = 'pred'
    output_name = 'pred'

    def init(self, recording):
        """
        num_dims: number of stimulus channels (y axis of STRF)
        num_coefs: number of temporal channels of STRF
        baseline: initial value of DC offset
        fit_fields: names of fitted parameters
        random: randomize initial values of fir coefficients
        """
        super().init(recording)
        if self.coefs is None:
            shape = self.input_channels, self.num_coefs
            if self.random_init:
                self.coefs = np.random.normal(loc=0.0, scale=0.0025,
                                              size=shape)
            else:
                self.coefs = np.zeros(shape)

        self.do_trial_plot = self.plot_fns[0]

    def evaluate(self, recording, mode):
        # Extract the Numpy array from the signal we need in the recording.
        x_signal = recording.get_signal(self.input_name)
        x = x_signal.as_continuous()

        # Do the computation. This returns a Numpy array.
        y = fir_filter(x, self.coefs, self.baseline, bank_count=self.bank_count)

        # Check if we need to normalize the data
        if self.norm_output:
            y = self.normalize(x, mode)

        y_signal = signal_like(x_signal, y, chans=['FIR'])
        return recording_like(recording, {self.output_name: y_signal})

    def get_strf(self):
        wc = self.parent_stack \
            .find_module('filters.weight_channels', self.output_name)
        w = wc.get_coefs()
        h = self.coefs
        return w.T @ h


################################################################################
# Short-term plasticity
################################################################################
class stp(Module):
    """
    stp - simulate short-term plasticity with the Tsodyks and Markram model

    m.editable_fields = {'num_channels', 'strength', 'tau', 'strength2', 'tau2',...
                    'per_channel', 'offset_in', 'facil_on', 'crosstalk',...
                    'input', 'input_mod','time', 'output' };
    """
    name='filters.stp'
    user_editable_fields=['input_name','output_name','fit_fields','num_channels','u','tau','offset_in','deponly','crosstalk']
    plot_fns=[nems.utilities.plot.pre_post_psth, nems.utilities.plot.plot_spectrogram, nems.utilities.plot.plot_stp]
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
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        if self.d_in and not(num_dims):
            num_dims = self.d_in[0][self.input_name].shape[0]
        Zmat = np.zeros([num_dims, num_channels])
        if not u:
            u = Zmat
        if not tau:
            tau = Zmat + 0.1
        if not offset_in:
            offset_in = Zmat

        self.num_dims = num_dims
        self.num_channels = num_channels
        self.fit_fields = fit_fields
        self.do_trial_plot = self.plot_fns[0]

        # stp parameters should be matrices num_dims X num_channels or 1 X num_channels,
        # and in the latter case be replicated across num_dims
        self.u = u
        self.tau = tau
        self.offset_in = offset_in
        self.crosstalk = crosstalk

    def my_eval(self, X):
        s = X.shape

        tstim = (X > 0) * X

        # TODO : enable crosstalk

        # TODO : for each stp channel, current just forcing 1
        Y = np.zeros([0, s[1], s[2]])
        di = np.ones(s)
        for j in range(0, self.num_channels):
            # force only depression, no facilitation
            ui = np.absolute(self.u[:, j])
            # ui=self.u[:,j]

            # convert tau units from sec to bins
            taui = np.absolute(self.tau[:, j]) * self.d_in[0]['fs']

            # go through each stimulus channel
            for i in range(0, s[0]):

                # limits, assumes input (X) range is approximately -1 to +1
                if ui[i] > 0.5:
                    ui[i] = 0.5
                elif ui[i] < -0.5:
                    ui[i] = -0.5
                if taui[i] < 0.5:
                    taui[i] = 0.5

                for tt in range(1, s[2]):
                    td = di[i, :, tt - 1]  # previous time bin depression
                    if ui[i] > 0:
                        delta = (1 - td) / taui[i] - \
                            ui[i] * td * tstim[i, :, tt - 1]
                        td = td + delta
                        td[td < 0] = 0
                    else:
                        delta = (1 - td) / taui[i] - \
                            ui[i] * td * tstim[i, :, tt - 1]
                        td = td + delta
                        td[td < 1] = 1
                    di[i, :, tt] = td

            Y = np.append(Y, di * X, 0)

        return Y

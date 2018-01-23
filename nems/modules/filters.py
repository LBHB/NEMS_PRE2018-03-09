import logging
log = logging.getLogger(__name__)

from nems.modules.base import nems_module as Module
import nems.utilities.utils

import numpy as np
from scipy import signal


################################################################################
# Channel weighting
################################################################################
def weight_channels_local(x, weights, baseline=None):
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
    #x = np.swapaxes(x, -3, -2)
    x = weights @ x
    #x = np.swapaxes(x, -3, -2)
    if baseline is not None:
        x += baseline[..., np.newaxis, np.newaxis]
    return x


class WeightChannels(Module):
    """
    weight_channels - apply a weighting matrix across a variable in the data
    stream. Used to provide spectral filters, directly imported from NARF.
    a helper function parm_fun can be defined to parameterize the weighting
    matrix. but by default the weights are each independent
    """
    name = 'filters.weight_channels'
    user_editable_fields = ['input_name', 'output_name', 'fit_fields',
                            'num_dims', 'num_chans', 'baseline', 'coefs',
                            'phi', 'parm_fun','norm_output']
    plot_fns = [nems.utilities.plot.plot_strf,
                nems.utilities.plot.plot_spectrogram]
    coefs = None
    num_chans = 1
    parm_fun = None
    parm_type = None
    
    def my_init(self, num_dims=0, num_chans=1, baseline=[[0]],
                fit_fields=None, parm_type=None, parm_fun=None, phi=[[0]],
                norm_output=False):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        if self.d_in and not(num_dims):
            num_dims = self.d_in[0][self.input_name].shape[0]
        self.num_dims = num_dims
        self.num_chans = num_chans
        self.baseline=np.ones([num_chans,1])*baseline
        self.fit_fields = fit_fields
        if parm_type:
            if parm_type == 'gauss':
                self.parm_fun = self.gauss_fn
                m = np.matrix(np.linspace(
                    1, self.num_dims, self.num_chans + 2))
                m = m[:, 1:-1] / self.num_dims
                s = np.ones([self.num_chans, 1]) / 4
                phi = np.concatenate([m.transpose(), s], 1)
            self.coefs = self.parm_fun(phi)
            if not fit_fields:
                self.fit_fields = ['phi']
        else:
            # self.coefs=np.ones([num_chans,num_dims])/num_dims/100
            #self.coefs = np.zeros([num_chans, num_dims])
            self.coefs = np.random.normal(
                1, 0.1, [num_chans, num_dims]) / num_dims
            if not fit_fields:
                self.fit_fields = ['coefs']
        self.parm_type = parm_type
        self.phi = np.array(phi)
        self.baseline = np.zeros(num_chans)

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

    def my_eval(self, x):
        # TODO: baseline was an option on this class; however, it was never
        # integrated in. In NARF, it was part of the fitting.
        if self.parm_fun:
            self.coefs = self.parm_fun(self.phi)
            coefs = self.coefs
        else:
            coefs = self.coefs
        if np.sum(np.abs(self.baseline[:]))>0:
            baseline=self.baseline
        else:
            baseline=None
            
        return weight_channels_local(x, coefs, baseline)

    def evaluate(self):
        del self.d_out[:]
        # create a copy of each input variable
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
            if self.output_name not in self.d_out[-1].keys():
                self.d_out[-1][self.output_name]=d[self.input_name]
                
        X=self.unpack_data(self.input_name,est=True)
        Z = self.my_eval(X)
        if self.norm_output:
            # compute std() of est data output and then normalize
            self.norm_factor=np.std(np.abs(Z),axis=1,keepdims=True)
            Z=Z/self.norm_factor
        self.pack_data(Z,self.output_name,est=True)
        
        if self.parent_stack.valmode:
            X=self.unpack_data(self.input_name,est=False)
            Z = self.my_eval(X)
            if self.norm_output:
                # don't recalc. just use factor that normalizes max of
                # estimation data to be 1
                Z=Z/self.norm_factor
            self.pack_data(Z,self.output_name,est=False)
        

class weight_channels(WeightChannels):
    pass

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


def fir_filter(x, coefficients, baseline=None, pad=False, bank_count=1):
    if pad:
        # TODO: This may become a moot option after Ivar's revamp of the data
        # loading system.
        pad_width = coefficients.shape[-1] * 2
        padding = [(0, 0)] * x.ndim
        padding[-1] = (0, pad_width)
        x = np.pad(x, padding, mode='constant')

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

    if pad:
        result = result[..., :-pad_width]

    if bank_count>1:
        # reshape inputs so that filter is summed separately across each bank
        # need to test this!
        s=list(result.shape)
        #print(s)
        ts0=np.int(s[-3]/bank_count)
        ts1=bank_count
        #print("{0},{1}".format(ts0,ts1))
        result=np.reshape(result,s[:-4]+[ts0,ts1]+s[-2:])
        result = np.sum(result, axis=-4)
    else:
        result = np.sum(result, axis=-3, keepdims=True)

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
    user_editable_fields = ['input_name','output_name','fit_fields',
                            'num_dims','num_coefs','coefs','baseline','random_init','bank_count']
    plot_fns = [nems.utilities.plot.plot_strf,
                nems.utilities.plot.plot_spectrogram]
    coefs = None
    baseline = np.zeros([1, 1])
    num_dims = 0
    random_init = False
    num_coefs = 20
    bank_count=1

    def my_init(self, num_dims=0, num_coefs=20, baseline=0, fit_fields=[
                'baseline', 'coefs'], random_init=False, coefs=None, bank_count=1):
        """
        num_dims: number of stimulus channels (y axis of STRF)
        num_coefs: number of temporal channels of STRF
        baseline: initial value of DC offset
        fit_fields: names of fitted parameters
        random: randomize initial values of fir coefficients
        """
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        if self.d_in and not(num_dims):
            num_dims = self.d_in[0][self.input_name].shape[0]
        self.num_dims = num_dims
        if bank_count<=1:
            self.bank_count=1
        else:
            self.bank_count=bank_count
        self.num_coefs = num_coefs
        self.baseline[0] = baseline
        self.random_init = random_init
        if coefs:
            self.coefs = coefs
        elif random_init is True:
            self.coefs = np.random.normal(
                loc=0.0, scale=0.0025, size=[num_dims, num_coefs])
        else:
            self.coefs = np.zeros([num_dims, num_coefs])
        self.fit_fields = fit_fields
        self.do_trial_plot = self.plot_fns[0]

    def my_eval_old(self, x):
        return fir_filter(x, self.coefs, self.baseline, bank_count=self.bank_count)

    def my_eval(self,X):
        s=X.shape
        X=np.reshape(X,[s[0],-1])
        for i in range(0,s[0]):
            y=np.convolve(X[i,:],self.coefs[i,:])
            X[i,:]=y[0:X.shape[1]]
            
        if self.bank_count:
            # reshape inputs so that filter is summed separately across each bank
            ts0=np.int(s[0]/self.bank_count)
            ts1=self.bank_count
            X=np.reshape(X,[ts1,ts0,-1])
            
        X=X.sum(1)+self.baseline
        s=list(s)
        s[0]=self.bank_count
        Y=np.reshape(X,s)
        return Y

    def get_strf(self):
        h = self.coefs

        # if weight channels exist and dimensionality matches, generate a full
        # STRF
        try:
            wcidx = nems.utilities.utils.find_modules(self.parent_stack, "filters.weight_channels")
            if len(wcidx) > 0 and self.parent_stack.modules[wcidx[0]].output_name == self.output_name:
                wcidx = wcidx[0]
            elif len(wcidx) > 1 and self.parent_stack.modules[wcidx[1]].output_name == self.output_name:
                wcidx = wcidx[1]
            else:
                wcidx = -1
        except BaseException:
            wcidx = -1

        if self.name == "filters.fir" and wcidx >= 0:
            w = self.parent_stack.modules[wcidx].coefs
            if w.shape[0] == h.shape[0]:
                h = np.matmul(w.transpose(), h)

        return h

class fir(FIR): #clone of FIR
    pass

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



class PsthModel(Module):
    name = 'filters.psthmodel'
    plot_fns = [nems.utilities.plot.sorted_raster,
                nems.utilities.plot.raster_plot]
    """
    Replaces stim with average resp for each stim (i.e., the PSTH). 
    This is the 'perfect' model
    used for comparing different models of pupil state gain.
    
    SVD added, pulled out of NS's pupil-specific analysis
    """

    def my_init(self):
        log.info('Replacing stimulus with averaged response raster')

    def evaluate(self):
        del self.d_out[:]
        # create a copy of pointer to each input variable
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        for f_in, f_out in zip(self.d_in, self.d_out):
            Xa = f_in['avgresp']
            R = f_in['replist']
            X = np.expand_dims(np.squeeze(Xa[R, :]),axis=0)
            f_out[self.output_name] = X
            
            

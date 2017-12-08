#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules that apply a filter to the stimulus


Created on Fri Aug  4 13:36:43 2017

@author: shofer
"""
from nems.modules.base import nems_module
import nems.utilities.utils

import numpy as np
from scipy import signal



class weight_channels(nems_module):
    """
    weight_channels - apply a weighting matrix across a variable in the data
    stream. Used to provide spectral filters, directly imported from NARF.
    a helper function parm_fun can be defined to parameterize the weighting
    matrix. but by default the weights are each independent
    """
    name='filters.weight_channels'
    user_editable_fields=['input_name','output_name','fit_fields','num_dims','num_chans','coefs','phi','parm_fun']
    plot_fns=[nems.utilities.plot.plot_strf,nems.utilities.plot.plot_spectrogram]
    coefs=None
    num_chans=1
    parm_fun=None
    parm_type=None

    def my_init(self, num_dims=0, num_chans=1, fit_fields=None, parm_type=None,
                parm_fun=None):
        # TODO: num_dims and num_chans need to be renamed. We should probably
        # revise this system a bit and use classmethod constructors to provide
        # different ways of initializing the class.
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        if self.d_in and not(num_dims):
            num_dims=self.d_in[0][self.input_name].shape[0]
        self.num_dims=num_dims
        self.num_chans=num_chans
        self.fit_fields=fit_fields
        if parm_type:
            if parm_type=='gauss':
                # TODO: This needs to be documented before I can adequately
                # refactor this.
                self.parm_fun=self.gauss_fn
                m=np.matrix(np.linspace(1,self.num_dims,self.num_chans+2))
                m=m[:,1:-1]/self.num_dims
                s=np.ones([self.num_chans,1])/4
                phi=np.concatenate([m.transpose(),s],1)
                self.phi=np.array(phi)
            if not fit_fields:
                self.fit_fields=['phi']
        else:
            #self.coefs=np.ones([num_chans,num_dims])/num_dims/100
            self.coefs=np.random.normal(1,0.1,[num_chans,num_dims])/num_dims
            if not fit_fields:
                self.fit_fields=['coefs']
        self.parm_type=parm_type

        self.y_offset = np.zeros(num_chans)

    def gauss_fn(self,phi):
        coefs=np.zeros([self.num_chans,self.num_dims])
        for i in range(0,self.num_chans):
            m=phi[i,0]*self.num_dims
            s=phi[i,1]*self.num_dims
            if s<0.05:
                s=0.05
            if (m<0 and m<s):
                s=-m
            elif (m>self.num_dims and m>self.num_dims+s):
                s=m-self.num_dims

            x=np.arange(0,self.num_dims)
            coefs[i,:]=np.exp(-np.square((x-m)/s))
            coefs[i,:]=coefs[i,:]/np.sum(coefs[i,:])
        return coefs

    def get_weights(self):
        # TODO: phi should always exist. If we just want a null transform, then
        # we'd define parm_fun as a null transform.
        try:
            coefs = self.parm_fun(self.phi)
        except (TypeError, AttributeError):
            coefs = self.coefs

        # Normalize so the weights are one
        coefs /= np.sum(coefs, axis=-1)[..., np.newaxis]
        return coefs

    def get_y_offset(self):
        return self.y_offset

    def my_eval(self, x):
        # We need to shift the channel dimension to the second-to-last dimension
        # so that matmul will work properly (it operates on the last two
        # dimensions of the inputs and treats the rest of the dimensions as
        # stacked matrices).
        weights = self.get_weights()
        y_offset = self.get_y_offset()
        x = np.swapaxes(x, -3, -2)
        x = weights @ x
        x = np.swapaxes(x, -3, -2)
        x += y_offset[..., np.newaxis, np.newaxis]
        return x


class fir(nems_module):
    """
    fir - the workhorse linear fir filter module. Takes in a 3D stim array
    (channels,stims,time), convolves with FIR coefficients, applies a baseline DC
    offset, and outputs a 2D stim array (stims,time).
    """
    name='filters.fir'
    user_editable_fields=['input_name','output_name','fit_fields','num_dims','num_coefs','coefs','baseline','random_init']
    plot_fns=[nems.utilities.plot.plot_strf, nems.utilities.plot.plot_spectrogram]
    coefs=None
    baseline=np.zeros([1,1])
    num_dims=0
    random_init=False
    num_coefs=20

    def my_init(self, num_dims=0, num_coefs=20, baseline=0, fit_fields=['baseline','coefs'],random_init=False, coefs=None):
        """
        num_dims: number of stimulus channels (y axis of STRF)
        num_coefs: number of temporal channels of STRF
        baseline: initial value of DC offset
        fit_fields: names of fitted parameters
        random: randomize initial values of fir coefficients
        """
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        if self.d_in and not(num_dims):
            num_dims=self.d_in[0][self.input_name].shape[0]
        self.num_dims=num_dims
        self.num_coefs=num_coefs
        self.baseline[0]=baseline
        self.random_init=random_init
        if coefs:
            self.coefs=coefs
        elif random_init is True:
            self.coefs=np.random.normal(loc=0.0,scale=0.0025,size=[num_dims,num_coefs])
        else:
            self.coefs=np.zeros([num_dims,num_coefs])
        self.fit_fields=fit_fields
        self.do_trial_plot=self.plot_fns[0]

    def get_coefs(self):
        return self.coefs

    def get_baseline(self):
        return self.baseline

    def get_zi(self, b, x):
        # This is the approach NARF uses. If the initial value of x[0] is 1,
        # this is identical to the NEMS approach. We need to provide zi to
        # lfilter to force it to return the final coefficients of the dummy
        # filter operation.
        n_taps = len(b)
        null_data = np.full(n_taps*2, x[0])
        zi = np.ones(n_taps-1)
        return signal.lfilter(b, [1], null_data, zi=zi)[1]

    def my_eval(self, x):
        coefs = self.get_coefs()
        baseline = self.get_baseline()

        # TODO: This will be a nice addition, but for now let's leave it out
        # because NARF doesn't do this.
        #pad_width = coefs.shape[-1] * 2
        #padding = [(0, 0)] * X.ndim
        #padding[-1] = (0, pad_width)
        #X = np.pad(X, padding, mode='constant')

        result = []
        for x, c in zip(x, coefs):
            old_shape = x.shape
            x = x.ravel()
            zi = self.get_zi(c, x)
            r, zf = signal.lfilter(c, [1], x, zi=zi)
            r.shape = old_shape
            result.append(r[np.newaxis])
        result = np.concatenate(result)
        return result.sum(axis=0) + baseline

    def get_strf(self):
        h=self.coefs

        # if weight channels exist and dimensionality matches, generate a full STRF
        try:
            wcidx=nems.utilities.utils.find_modules(self.parent_stack,"filters.weight_channels")
            if len(wcidx)>0 and self.parent_stack.modules[wcidx[0]].output_name==self.output_name:
                wcidx=wcidx[0]
            elif len(wcidx)>1 and self.parent_stack.modules[wcidx[1]].output_name==self.output_name:
                wcidx=wcidx[1]
            else:
                wcidx=-1
        except:
            wcidx=-1

        if self.name=="filters.fir" and wcidx>=0:
            #print(m.name)
            w=self.parent_stack.modules[wcidx].coefs
            if w.shape[0]==h.shape[0]:
                h=np.matmul(w.transpose(), h)

        return h


class stp(nems_module):
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



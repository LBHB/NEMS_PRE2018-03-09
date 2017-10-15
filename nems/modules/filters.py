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



class weight_channels(nems_module):
    """
    weight_channels - apply a weighting matrix across a variable in the data
    stream. Used to provide spectral filters, directly imported from NARF.
    a helper function parm_fun can be defined to parameterize the weighting
    matrix. but by default the weights are each independent
    """
    name='filters.weight_channels'
    user_editable_fields=['input_name','output_name','fit_fields','num_dims','num_chans','baseline','coefs','phi','parm_fun']
    plot_fns=[nems.utilities.plot.plot_strf,nems.utilities.plot.plot_spectrogram]
    coefs=None
    baseline=np.zeros([1,1])
    num_chans=1
    parm_fun=None
    parm_type=None
    def my_init(self, num_dims=0, num_chans=1, baseline=[[0]], 
                fit_fields=None, parm_type=None, parm_fun=None, phi=[[0]]):
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        if self.d_in and not(num_dims):
            num_dims=self.d_in[0][self.input_name].shape[0]
        self.num_dims=num_dims
        self.num_chans=num_chans
        self.baseline=np.array(baseline)
        self.fit_fields=fit_fields
        if parm_type:
            if parm_type=='gauss':
                self.parm_fun=self.gauss_fn
                m=np.matrix(np.linspace(1,self.num_dims,self.num_chans+2))
                m=m[:,1:-1]/10
                s=np.ones([self.num_chans,1])*4/10
                phi=np.concatenate([m.transpose(),s],1)
            self.coefs=self.parm_fun(phi)
            if not fit_fields:
                self.fit_fields=['phi']
        else:
            #self.coefs=np.ones([num_chans,num_dims])/num_dims/100
            self.coefs=np.random.normal(1,0.1,[num_chans,num_dims])/num_dims
            if not fit_fields:
                self.fit_fields=['coefs']
        self.parm_type=parm_type
        self.phi=np.array(phi)
    
    def gauss_fn(self,phi):
        coefs=np.zeros([self.num_chans,self.num_dims])
        for i in range(0,self.num_chans):
            m=phi[i,0]*10
            s=phi[i,1]*10
            x=np.arange(0,self.num_dims)
            coefs[i,:]=np.exp(-np.square((x-m)/s))
            coefs[i,:]=coefs[i,:]/np.sum(coefs[i,:])
        return coefs
        
    def my_eval(self,X):
        #if not self.d_out:
        #    # only allocate memory once, the first time evaling. rish is that output_name could change
        if self.parm_fun:
            self.coefs=self.parm_fun(self.phi)
        s=X.shape
        X=np.reshape(X,[s[0],-1])
        X=np.matmul(self.coefs,X)
        s=list(s)
        s[0]=self.num_chans
        Y=np.reshape(X,s)
        return Y
    
 
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
        
    def my_eval(self,X):
        s=X.shape
        X=np.reshape(X,[s[0],-1])
        for i in range(0,s[0]):
            y=np.convolve(X[i,:],self.coefs[i,:])
            X[i,:]=y[0:X.shape[1]]
        X=X.sum(0)+self.baseline
        Y=np.reshape(X,s[1:])
        return Y
    
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
            ui=np.absolute(self.u[:,j])
            #ui=self.u[:,j]
            taui=self.tau[:,j]*self.d_in[0]['respFs']  # norm by sampling rate so that tau is in units of sec
            
            # go through each stimulus channel
            for i in range(0,s[0]):
                
                # limits:
                if ui[i]>0.5:
                    ui[i]=0.5
                elif ui[i]<-0.5:
                    ui[i]=-0.5
                    
                if taui[i]<0.001:
                    taui[i]=0.001
                    
                for tt in range(1,s[2]):
                    td=di[i,:,tt-1]
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
    
        
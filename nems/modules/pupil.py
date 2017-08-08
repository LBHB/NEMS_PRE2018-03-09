#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules for manipulating pupil data


Created on Fri Aug  4 13:29:30 2017

@author: shofer
"""
from nems.modules.base import nems_module
import nems.utilities.utils as nu

import numpy as np
import copy 
import scipy.special as sx

class model(nems_module):
    name='pupil_model'
    plot_fns=[nu.sorted_raster,nu.raster_plot]
    """
    Replaces stim with average resp for each stim. This is the 'perfect' model
    used for comparing different models of pupil state gain.
    """
    
    def evaluate(self,nest=0):
        if nest==0:
            del self.d_out[:]
            for i,val in enumerate(self.d_in):
                self.d_out.append(copy.deepcopy(val))
        for f_in,f_out in zip(self.d_in,self.d_out):
            Xa=f_in['avgresp']
            if f_in['est'] is False:
                R=f_in['replist'][nest]
                X=np.zeros(f_in['resp'][nest].shape)
                for i in range(0,R.shape[0]):
                    X[i,:]=Xa[R[i],:]
                f_out['stim'][nest]=X
            else:
                R=f_in['replist']
                X=np.zeros(f_in['resp'].shape)
                for i in range(0,R.shape[0]):
                    X[i,:]=Xa[R[i],:]
                f_out['stim']=X
                
class pupgain(nems_module): 
    """
    state_gain - apply a gain/offset based on continuous pupil diameter, or some 
    other continuous variable. Does not use standard my_eval, instead uses its own
    evaluate() that overrides the nems_module evaluate()
    
    @author: shofer
    """
    #Changed to helper function based general module --njs June 29 2017
    name='pupgain'
    plot_fns=[nu.pred_act_scatter_smooth,nu.pre_post_psth,nu.pred_act_psth_all,nu.non_plot]
    
    def my_init(self,d_in=None,gain_type='linpupgain',fit_fields=['theta'],theta=[0,1,0,0],premodel=False,
                order=None):
        if premodel is True:
            self.do_plot=self.plot_fns[1]
        #self.linpupgain=np.zeros([1,4])
        #self.linpupgain[0][1]=0
        self.fit_fields=fit_fields
        self.gain_type=gain_type
        theta=np.array([theta])
        self.theta=theta
        self.order=order
        self.do_plot=self.plot_fns[0]
        #self.data_setup(d_in)
        print('state_gain parameters created')
        
    def nopupgain_fn(self,X,Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve 
        state variable. This is the "control" for the state_gain exploration.
        """
        Y=self.theta[0,0]+self.theta[0,1]*X
        return(Y)   
    def linpupgain_fn(self,X,Xp):
        Y=self.theta[0,0]+(self.theta[0,2]*Xp)+(self.theta[0,1]*X)+self.theta[0,3]*np.multiply(Xp,X)
        return(Y)
    def exppupgain_fn(self,X,Xp):
        Y=self.theta[0,0]+self.theta[0,1]*X*np.exp(self.theta[0,2]*Xp+self.theta[0,3])
        return(Y)
    def logpupgain_fn(self,X,Xp):
        Y=self.theta[0,0]+self.theta[0,1]*X*np.log(self.theta[0,2]+Xp+self.theta[0,3])
        return(Y)
    def polypupgain_fn(self,X,Xp):
        """
        Fits a polynomial gain function: 
        Y = g0 + g*X + d1*X*Xp^1 + d2*X*Xp^2 + ... + d(n-1)*X*Xp^(n-1) + dn*X*Xp^n
        """
        deg=self.theta.shape[1]
        Y=0
        for i in range(0,deg-2):
            Y+=self.theta[0,i]*X*np.power(Xp,i+1)
        Y+=self.theta[0,-2]+self.theta[0,-1]*X
        return(Y)
    def powerpupgain_fn(self,X,Xp):
        """
        Slightly different than polypugain. Y = g0 + g*X + d0*Xp^n + d*X*Xp^n
        """
        deg=self.order
        v=self.theta
        Y=v[0,0] + v[0,1]*X + v[0,2]*np.power(Xp,deg) + v[0,3]*np.multiply(X,np.power(Xp,deg))
        return(Y)
    def Poissonpupgain_fn(self,X,Xp): #Kinda useless, might delete ---njs
        u=self.theta[0,1]
        Y=self.theta[0,0]*X*np.divide(np.exp(-u)*np.power(u,Xp),sx.factorial(Xp))
        return(Y)
    def butterworthHP_fn(self,X,Xp):
        """
        Applies a Butterworth high pass filter to the pupil data, with a DC offset.
        Pupil diameter is treated here as analogous to frequency, and the fitted 
        parameters are DC offset, overall gain, and f3dB. Order is specified, and
        controls how fast the rolloff is.
        """
        n=self.order
        Y=self.theta[0,2]+self.theta[0,0]*X*np.divide(np.power(np.divide(Xp,self.theta[0,1]),n),
                    np.sqrt(1+np.power(np.divide(Xp,self.theta[0,1]),2*n)))
        return(Y)
              
    def evaluate(self,nest=0):
        if nest==0:
            del self.d_out[:]
            for i,val in enumerate(self.d_in):
                self.d_out.append(copy.deepcopy(val))
        for f_in,f_out in zip(self.d_in,self.d_out):
            if f_in['est'] is False:
                X=copy.deepcopy(f_in[self.input_name][nest])
                Xp=copy.deepcopy(f_in[self.state_var][nest])
                Z=getattr(self,self.gain_type+'_fn')(X,Xp)
                f_out[self.output_name][nest]=Z
            else:
                X=copy.deepcopy(f_in[self.input_name])
                Xp=copy.deepcopy(f_in[self.state_var])
                Z=getattr(self,self.gain_type+'_fn')(X,Xp)
                f_out[self.output_name]=Z
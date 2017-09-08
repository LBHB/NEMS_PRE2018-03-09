#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules for manipulating pupil data


Created on Fri Aug  4 13:29:30 2017

@author: shofer
"""
from nems.modules.base import nems_module
import nems.utilities.utils
import nems.utilities.plot

import numpy as np
import copy 
import scipy.special as sx

class model(nems_module):
    name='pupil.model'
    plot_fns=[nems.utilities.plot.sorted_raster,nems.utilities.plot.raster_plot]
    """
    Replaces stim with average resp for each stim. This is the 'perfect' model
    used for comparing different models of pupil state gain.
    """
    def my_init(self):
        print('Replacing stimulus with averaged response raster')
        self.field_dict=locals()
        self.field_dict.pop('self',None)
    
    def evaluate(self,nest=0):
        if nest==0:
            del self.d_out[:]
            for i,val in enumerate(self.d_in):
                self.d_out.append(copy.deepcopy(val))
        for f_in,f_out in zip(self.d_in,self.d_out):
            Xa=f_in['avgresp']
            if f_in['est'] is False and self.parent_stack.nests>0:
                R=f_in['replist'][nest]
                X=np.squeeze(Xa[R,:])
                #X=np.zeros(f_in['resp'][nest].shape)
                #for i in range(0,R.shape[0]):
                #    X[i,:]=Xa[R[i],:]
                f_out['stim'][nest]=X
            else:
                R=f_in['replist']
                X=np.squeeze(Xa[R,:])
                #X=np.zeros(f_in['resp'].shape)
                #for i in range(0,R.shape[0]):
                #    X[i,:]=Xa[R[i],:]
                f_out['stim']=X
                
class pupgain(nems_module): 
    """
    state_gain - apply a gain/offset based on continuous pupil diameter, or some 
    other continuous variable. Does not use standard my_eval, instead uses its own
    evaluate() that overrides the nems_module evaluate()
    
    @author: shofer
    """
    #Changed to helper function based general module --njs June 29 2017
    name='pupil.pupgain'
    user_editable_fields = ['input_name','output_name','fit_fields','state_var','gain_type','theta']
    gain_type='linpupgain'
    plot_fns=[nems.utilities.plot.state_act_scatter_smooth,nems.utilities.plot.pre_post_psth,nems.utilities.plot.pred_act_psth_all,nems.utilities.plot.non_plot]
    
    def my_init(self,input_name="stim",output_name="stim",state_var="pupil",
                gain_type='linpupgain',fit_fields=['theta'],theta=[0,1,0,0],
                order=None):
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.fit_fields=fit_fields
        self.gain_type=gain_type
        self.theta=np.array([theta])
        self.order=order
        self.do_plot=self.plot_fns[0]
        
    def nopupgain_fn(self,X,Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve 
        state variable. This is the "control" for the state_gain exploration.
        """
        Y=self.theta[0,0]+self.theta[0,1]*X
        return(Y)   
    def linpupgainctl_fn(self,X,Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve 
        state variable. This is the "control" for the state_gain exploration.
        
        SVD mod: shuffle pupil, keep same number of parameters for proper control
        """
        
        # OLD WAY -- not random enough
        if 0:
            s=Xp.shape
            n=np.int(np.ceil(s[0]/2))
            #print(s)
            #print(n)
            #print(Xp.shape)
            Xp=np.roll(Xp,n,0)
        else:
            # save current random state
            prng = np.random.RandomState()
            save_state = prng.get_state()
            prng = np.random.RandomState(1234567890)
            
            # shuffle state vector across trials (time)
            prng.shuffle(Xp)
            
            # restore saved random state
            prng.set_state(save_state)
        
        Y,Xp=self.linpupgain_fn(X,Xp)
        return Y,Xp
        
    def linpupgain_fn(self,X,Xp):
        Y=self.theta[0,0]+(self.theta[0,2]*Xp)+(self.theta[0,1]*X)+self.theta[0,3]*np.multiply(Xp,X)
        return Y,Xp
        
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
                Z,Xp=getattr(self,self.gain_type+'_fn')(X,Xp)
                f_out[self.output_name][nest]=Z
                f_out[self.state_var][nest]=Xp
            else:
                X=copy.deepcopy(f_in[self.input_name])
                Xp=copy.deepcopy(f_in[self.state_var])
                Z,Xp=getattr(self,self.gain_type+'_fn')(X,Xp)
                f_out[self.output_name]=Z
                f_out[self.state_var]=Xp
                
                
class state_weight(nems_module): 
    """
    pupweight - combined weighting of two predicted PSTHs, depending on state_var
    @author: svd
    """

    name='pupil.state_weight'
    user_editable_fields = ['input_name','output_name','fit_fields','state_var','input_name2','weight_type','theta']
    weight_type='linear'
    plot_fns=[nems.utilities.plot.state_act_scatter_smooth,nems.utilities.plot.pre_post_psth,nems.utilities.plot.pred_act_psth_all,nems.utilities.plot.non_plot]
    input_name2='stim2'
    state_var='pupil'
    theta=np.zeros([1,2])
    def my_init(self,input_name="stim",input_name2="stim2",state_var="pupil",
                weight_type='linear',fit_fields=['theta'],theta=[0,0.001]):
        self.input_name=input_name
        self.input_name2=input_name2
        self.state_var=state_var
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.fit_fields=fit_fields
        self.weight_type=weight_type
        self.my_eval=getattr(self,self.weight_type+'_fn')
        self.theta=np.array([theta])
        self.do_plot=self.plot_fns[0]
        
    def linear_fn(self,X1,X2,Xp):
        """
        linear weighting of two predicted PSTHs, depending on state_var
        w= a + b * p(t)  hard bounded at 0 and 1 
        """
        w=self.theta[0,0]+self.theta[0,1]*Xp
        w[w<0]=0
        w[w>1]=1
        Y=(1-w)*X1+w*X2
        return(Y,Xp)
    
    def linearctl_fn(self,X1,X2,Xp):
        """
        shuffle pupil, keep same number of parameters for proper control
        """
        
        # save current random state
        prng = np.random.RandomState()
        save_state = prng.get_state()
        prng = np.random.RandomState(1234567890)
        
        # shuffle state vector across trials (time)
        prng.shuffle(Xp)
        
        # restore saved random state
        prng.set_state(save_state)
        
        #s=Xp.shape
        #n=np.int(np.ceil(s[0]/2))
        #Xp=np.roll(Xp,n,0)
        
        Y,Xp=self.linear_fn(X1,X2,Xp)
        
        
        return(Y,Xp)
              
    def evaluate(self,nest=0):
        if nest==0:
            del self.d_out[:]
            for i,val in enumerate(self.d_in):
                self.d_out.append(copy.deepcopy(val))
        for f_in,f_out in zip(self.d_in,self.d_out):
            if f_in['est'] is False:
                X1=copy.deepcopy(f_in[self.input_name][nest])
                X2=copy.deepcopy(f_in[self.input_name2][nest])
                Xp=copy.deepcopy(f_in[self.state_var][nest])
                Y,Xp=self.my_eval(X1,X2,Xp)
                f_out[self.output_name][nest]=Y
                f_out[self.state_var][nest]=Xp
            else:
                X1=copy.deepcopy(f_in[self.input_name])
                X2=copy.deepcopy(f_in[self.input_name2])
                Xp=copy.deepcopy(f_in[self.state_var])
                Y,Xp=self.my_eval(X1,X2,Xp)
                f_out[self.output_name]=Y
                f_out[self.state_var]=Xp
                
                
             
                
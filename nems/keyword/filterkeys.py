#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 10:37:57 2017

@author: shofer
"""

import nems.modules as nm
from nems.utilities.utils import mini_fit

def wc01(stack):
    """
    Applies a 1 channel spectral filter matrix to the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=1)

def wc02(stack):
    """
    Applies a 2 channel spectral filter matrix to the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=2)

def wc03(stack):
    """
    Applies a 3 channel spectral filter matrix to the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=3)

def wc04(stack):
    """
    Applies a 4 channel spectral filter matrix to a the data
    stream.
    """
    stack.append(nm.filters.weight_channels,num_chans=4)

def wcg01(stack):
    """
    Applies a 1 channel spectral filter matrix to the data stream.
    Each channel constrained to be a Gaussian with 2 parameters (mu,sigma).
    """
    stack.append(nm.filters.weight_channels,num_chans=1,parm_type="gauss")

def wcg02(stack):
    """
    Applies a 2 channel spectral filter matrix to the data stream.
    Each channel constrained to be a Gaussian with 2 parameters (mu,sigma).
    """
    stack.append(nm.filters.weight_channels,num_chans=2,parm_type="gauss")

def wcg03(stack):
    """
    Applies a 3 channel spectral filter matrix to the data stream.
    Each channel constrained to be a Gaussian with 2 parameters (mu,sigma).
    """
    stack.append(nm.filters.weight_channels,num_chans=3,parm_type="gauss")

def wcg04(stack):
    """
    Applies a 4 channel spectral filter matrix to the data stream.
    Each channel constrained to be a Gaussian with 2 parameters (mu,sigma).
    """
    stack.append(nm.filters.weight_channels,num_chans=4,parm_type="gauss")

def fir10(stack):
    """
    Appends a 10 temporal bin finite impluse response (FIR) filter to the datastream. 
    This filter can serve as either the entire STRF for the cell and be fitted as such, 
    or as the temporal filter in the factorized STRF if used in conjuction with the 
    weight channel spectral filter. 
    
    This keyword initializes the FIR coefficients to 0, and performs a fit on the 
    FIR coefficients (and weight channel coefficients, if a weight channel matrix
    is included in the model).
    """
    stack.append(nm.filters.fir,num_coefs=10)
    mini_fit(stack,mods=['filters.weight_channels','filters.fir','filters.stp'])
    
def fir10r(stack):
    """
    Appends a 10 temporal bin finite impluse response (FIR) filter to the datastream. 
    This filter can serve as either the entire STRF for the cell and be fitted as such, 
    or as the temporal filter in the factorized STRF if used in conjuction with the 
    weight channel spectral filter. 
    
    This keyword draws the intial FIR coeffcients from a normal distribution about 0 
    with a standard distribution of 0.0025, and performs a fit on the FIR coefficients 
    (and weight channel coefficients, if a weight channel matrix is included in the model).
    """
    stack.append(nm.filters.fir,num_coefs=10,random=True)
    mini_fit(stack,mods=['filters.weight_channels','filters.fir','filters.stp'])
    
def fir15(stack):
    """
    Appends a 15 temporal bin finite impluse response (FIR) filter to the datastream. 
    This filter can serve as either the entire STRF for the cell and be fitted as such, 
    or as the temporal filter in the factorized STRF if used in conjuction with the 
    weight channel spectral filter. 
    
    This keyword initializes the FIR coefficients to 0, and performs a fit on the 
    FIR coefficients (and weight channel coefficients, if a weight channel matrix
    is included in the model).
    """
    stack.append(nm.filters.fir,num_coefs=15)
    mini_fit(stack,mods=['filters.weight_channels','filters.fir','filters.stp'])

def fir20(stack):
    """
    Appends a 20 temporal bin finite impluse response (FIR) filter to the datastream. 
    This filter can serve as either the entire STRF for the cell and be fitted as such, 
    or as the temporal filter in the factorized STRF if used in conjuction with the 
    weight channel spectral filter. 
    
    This keyword initializes the FIR coefficients to 0, and performs a fit on the 
    FIR coefficients (and weight channel coefficients, if a weight channel matrix
    is included in the model). -- for fir20, temporarily disabling stp in mini_fit
    """
    stack.append(nm.filters.fir,num_coefs=20)
    mini_fit(stack,mods=['filters.weight_channels','filters.fir'])

def stp1pc(stack):
    #stack.append(nm.aux.normalize)
    #stack.append(nm.filters.stp,num_channels=1,fit_fields=[])
    stack.append(nm.filters.stp,num_channels=1)
    stack.modules[-1].u[:]=0.01
    
def stp1pcon(stack):
    #stack.append(nm.aux.normalize)
    #stack.append(nm.filters.stp,num_channels=1,fit_fields=[])
    stack.append(nm.filters.stp,num_channels=1)
    stack.modules[-1].u[:]=0.1
    stack.modules[-1].tau[:]=0.5
   
    
def stp2pc(stack):
    stack.append(nm.filters.stp,num_channels=2)
    stack.modules[-1].u[:,0]=0.01
    stack.modules[-1].u[:,1]=0.1

def stp1pcn(stack):
    #stack.append(nm.aux.normalize)
    #stack.append(nm.filters.stp,num_channels=1,fit_fields=[])
    stack.append(nm.aux.normalize)
    stack.append(nm.filters.stp,num_channels=1)
    stack.modules[-1].u[:]=0.01


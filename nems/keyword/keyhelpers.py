#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword helper functions

Created on Fri Aug 11 10:39:17 2017

@author: shofer
"""
import nems.modules as nm
import nems.fitters as nf
import nems.utilities as ut
import numpy as np




def mini_fit(stack,mods=['filters.weight_channels','filters.fir','filters.stp']):
    """
    Helper function that module coefficients in mod list prior to fitting 
    all the model coefficients. This is often helpful, as it gets the model in the
    right ballpark before fitting other model parameters, especially when nonlinearities
    are included in the model.
    
    This function is not appended directly to the stack, but instead is included
    in keywords
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    stack.fitter=nf.fitters.basic_min(stack)
    stack.fitter.tolerance=0.0001
    fitidx=[]
    for i in mods:
        try:
            fitidx=fitidx+ut.utils.find_modules(stack,i)
        except:
            fitidx=fitidx+[]
    stack.fitter.fit_modules=fitidx
    
    stack.fitter.do_fit()
    stack.popmodule()
    
    
def create_parmlist(stack):
    """
    Helper function that assigns all fitted parameters for a model to a single (n,)
    phi vector and accociates it to the stack.parm_fits object
    """
    phi=[] 
    for idx,m in enumerate(stack.modules):
        this_phi=m.parms2phi()
        if this_phi.size:
            if stack.cv_counter==0:
                stack.fitted_modules.append(idx)
            phi.append(this_phi)
    phi=np.concatenate(phi)
    stack.parm_fits.append(phi)

        
        
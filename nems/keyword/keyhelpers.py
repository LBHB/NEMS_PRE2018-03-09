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

import pkgutil as pk



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
    fitidx=[]
    for i in mods:
        try:
            fitidx=fitidx+ut.utils.find_modules(stack,i)
        except:
            fitidx=fitidx+[]
    stack.fitter=nf.fitters.basic_min(stack,fit_modules=fitidx,tolerance=0.0001)
    
    stack.fitter.do_fit()
    stack.popmodule()
    
    
def create_parmlist(stack):
    """
    Helper function that assigns all fitted parameters for a model to a single (n,)
    phi vector and accociates it to the stack.parm_fits object
    """
    stack.fitted_modules=[]
    phi=[] 
    for idx,m in enumerate(stack.modules):
        this_phi=m.parms2phi()
        if this_phi.size:
            stack.fitted_modules.append(idx)
            phi.append(this_phi)
    phi=np.concatenate(phi)
    stack.parm_fits.append(phi)

# Nested Crossval
###############################################################################

def nested20(stack):
    """
    Keyword for 20-fold nested crossvalidation. Uses 5% validation chunks. 
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=20
    #stack.valfrac=0.05
    nest_helper(stack)
        
def nested10(stack):
    """
    Keyword for 10-fold nested crossvalidation. Uses 10% validation chunks.
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=10
    #stack.valfrac=0.1
    nest_helper(stack)
    
def nested5(stack):
    """
    Keyword for 5-fold nested crossvalidation. Uses 10% validation chunks.
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=5
    #stack.valfrac=0.2
    nest_helper(stack)

def nested2(stack):
    """
    Keyword for 2-fold nested crossvalidation. Uses 10% validation chunks.
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=2
    #stack.valfrac=0.5
    nest_helper(stack)

# Helper/Support Functions
###############################################################################

def nest_helper(stack):
    """
    Helper function for implementing nested crossvalidation. Essentially sets up
    a loop with the estimation part of fit_single_model inside. 
    """
    stack.cond=False
    cv_counter=0  # reset to avoid problem with val stage
    while cv_counter<stack.nests:
        print('Nest #'+str(cv_counter))
        stack.clear()
        stack.valmode=False
        for i in range(0,len(stack.keywords)-1):
            stack.keyfuns[stack.keywords[i]](stack)
            if stack.modules[-1].name=="est_val.crossval":
                stack.modules[-1].cv_counter=cv_counter
                stack.modules[-1].evaluate()
        cv_counter+=1

    #stack.cv_counter=0  # reset to avoid problem with val stage
        
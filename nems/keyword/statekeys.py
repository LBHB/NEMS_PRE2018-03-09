#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State keywords: both pupil gain and pupil model

Created on Fri Aug 11 11:08:54 2017

@author: shofer
"""

import logging
log = logging.getLogger(__name__)

import nems.modules as nm
from nems.utilities.utils import mini_fit
import nems.utilities as nu

from .registry import keyword_registry


# State Model keywords
###############################################################################

def psthpred(stack):
    """ keyword to "predict" the single-trial response based on the average
    (PSTH) response to that stimulus. Can be fed into pupil/behavior
    models
    """
    stack.append(nm.aux.psth)
    stack.modules[-1].do_plot=stack.modules[-1].plot_fns[2]  # psth
    
# Pupil Gain keywords
###############################################################################


def stategainctl(stack):
    """
    Applies a DC gain function entry-by-entry to the datastream:
        y = v1 + v2*x + <randomly shuffled pupil dc-gain>
    where x is the input matrix and v1,v2 are fitted parameters applied to
    each matrix entry (the same across all entries)
    """
    if stack.data[-1][0]['state'].shape[0]==4:
        theta0=[0,1,0,0,0,0,0,0,0,0]
    elif stack.data[-1][0]['state'].shape[0]==3:
        theta0=[0,1,0,0,0,0,0,0]
    elif stack.data[-1][0]['state'].shape[0]==2:
        theta0=[0,1,0,0,0,0]
    else:
        theta0=[0,1,0,0]
    stack.append(nm.state.state_gain,gain_type='lingainctl',state_var='state',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)

def stategain(stack):
    """
    Applies a linear pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*p + v4*x*p
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    if stack.data[-1][0]['state'].shape[0]==4:
        theta0=[0,1,0,0,0,0,0,0,0,0]
    elif stack.data[-1][0]['state'].shape[0]==3:
        theta0=[0,1,0,0,0,0,0,0]
    elif stack.data[-1][0]['state'].shape[0]==2:
        theta0=[0,1,0,0,0,0]
    else:
        theta0=[0,1,0,0]
    stack.append(nm.state.state_gain,gain_type='lingain',state_var='state',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)

def pupgainctl(stack):
    """
    Applies a DC gain function entry-by-entry to the datastream:
        y = v1 + v2*x + <randomly shuffled pupil dc-gain>
    where x is the input matrix and v1,v2 are fitted parameters applied to
    each matrix entry (the same across all entries)
    
    Only uses first pupil variable (raw pupil, not derivatives)
    """
    
    theta0=[0,1,0,0]
    stack.append(nm.state.state_gain,gain_type='lingainctl',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)

def pupgain(stack):
    """
    Applies a linear pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*p + v4*x*p
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2
    are fitted parameters applied to each matrix entry (the same across all entries)
    
    Only uses first pupil variable (raw pupil, not derivatives)
    """
    
    theta0=[0,1,0,0]
    stack.append(nm.state.state_gain,gain_type='lingain',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)

def pupdgainctl(stack):
    """
    Applies a DC gain function entry-by-entry to the datastream:
        y = v1 + v2*x + <randomly shuffled pupil dc-gain>
    where x is the input matrix and v1,v2 are fitted parameters applied to
    each matrix entry (the same across all entries)
    
    Uses all three pupil variable (raw pupil, pos+neg derivatives)
    """
    
    theta0=[0,1,0,0,0,0,0,0]
    stack.append(nm.state.state_gain,gain_type='lingainctl',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)

def pupdgain(stack):
    """
    Applies a linear pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*p + v4*x*p
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2
    are fitted parameters applied to each matrix entry (the same across all entries)
    
    Uses all three pupil variable (raw pupil, pos+neg derivatives)
    """
    
    theta0=[0,1,0,0,0,0,0,0]
    stack.append(nm.state.state_gain,gain_type='lingain',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)


def behgainctl(stack):
    """
    Applies a DC gain function entry-by-entry to the datastream:
        y = v1 + v2*x + <randomly shuffled pupil dc-gain>
    where x is the input matrix and v1,v2 are fitted parameters applied to
    each matrix entry (the same across all entries)
    """
    theta0=[0,1,0,0]
    stack.append(nm.state.state_gain,gain_type='lingainctl',state_var='behavior_condition',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)

def behgain(stack):
    """
    Applies a linear pupil gain function entry-by-entry to the datastream:
        y = v1 + v2*x + v3*p + v4*x*p
    where x is the input matrix, p is the matrix of pupil diameters, and v1,v2
    are fitted parameters applied to each matrix entry (the same across all entries)
    """
    theta0=[0,1,0,0]
    stack.append(nm.state.state_gain,gain_type='lingain',state_var='behavior_condition',fit_fields=['theta'],theta=theta0)
    mini_fit(stack,mods=['state.state_gain'])
    log.info(stack.modules[-1].theta)


# weighted combination 2 PSTHS usign pupil

def pupwgt(stack,weight_type='linear'):
    """
    linear weighted sum of stim and stim2, determined by pupil
        w=(phi[0]+phi[1] * p)
        Y=stim1 * (1-w) + stim2 * w
    hard bound on w  in (0,1)
    """
    # find fir and duplicate
    wtidx=nu.utils.find_modules(stack,'filters.weight_channels')
    firidx=nu.utils.find_modules(stack,'filters.fir')
    wtidx=wtidx[0]
    firidx=firidx[0]
    num_chans=stack.modules[wtidx].num_chans
    wcoefs=stack.modules[wtidx].coefs
    parm_type=stack.modules[wtidx].parm_type
    phi=stack.modules[wtidx].phi
    num_coefs=stack.modules[firidx].num_coefs
    coefs=stack.modules[firidx].coefs
    baseline=stack.modules[firidx].baseline
    stack.modules[wtidx].output_name='pred1'
    for ii in range(wtidx+1,firidx+1):
        stack.modules[ii].input_name='pred1'
        stack.modules[ii].output_name='pred1'
    stack.evaluate(wtidx)
    stack.append(nm.filters.weight_channels,output_name="pred2",num_chans=num_chans,phi=phi,parm_type=parm_type)
    stack.modules[-1].phi=phi
    stack.modules[-1].wcoefs=wcoefs
    stack.append(nm.aux.normalize,input_name="pred2",output_name="pred2")    
    stack.append(nm.filters.fir,num_coefs=num_coefs,input_name="pred2",output_name="pred2")
    stack.modules[-1].coefs=coefs*0.99
    stack.modules[-1].baseline=baseline*0.99

    stack.append(nm.state.state_weight,input_name="pred1",input_name2="pred2",
                 state_var="pupil",weight_type=weight_type,fit_fields=['theta'],theta=[0,0.01])
    stack.evaluate(wtidx)

    #mini_fit(stack,mods=['pupil.state_weight'])

def pupwgtctl(stack):
    """
    linear weighted sum of stim and stim2, determined by shuffled pupil
        w=(phi[0]+phi[1] * p_shuff)
        Y=stim1 * (1-w) + stim2 * w
    hard bound on w  in (0,1)
    """

    # call pupwgt with different weight_type
    statewgt(stack,weight_type='linearctl')


matches = ['state','psthpred','beh','pup']

for k, v in list(locals().items()):
    # TODO: this is a hack for now.
    for m in matches:
        if k.startswith(m):
            keyword_registry[k] = v
            continue

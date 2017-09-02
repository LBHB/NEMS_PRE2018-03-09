#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
User defined module keywords

Created on Fri Aug 11 11:18:34 2017

@author: shofer
"""

import nems.modules as nm
import nems.modules.user_def as ud



def jitterload(stack):
    """
    Loads a 2-channel, 1000 Hz .mat file containing data for Mateo's jitter tone
    project, and downsamples to 500 Hz. Applies a 5% estimation/validation split.
    """
    filepath='/auto/users/shofer/data/batch296mateo/'+str(stack.meta['cellid'])+'_b'+str(stack.meta['batch'])+'_envelope_fs1000.mat'
    print("Initializing load_mat with file {0}".format(filepath))
    stack.append(ud.load_baphy_ssa.load_baphy_ssa,file=filepath,fs=500)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def simpledemo00(stack):
    """
    Keyword for the simple_demo DC gain module. Appends the simple_demo module
    with its default arguments.
    
    Applies a simple DC gain and offset to the input data:
        y = v1*x + v2
    where x is the input variable, and v1,v2 are fitted parameters. 
    
    """
    stack.append(ud.demo.simple_demo)
    
def simpledemo01(stack):
    """
    Keyword for the simple_demo DC gain module. Uses arguments different than the
    default.
    
    Applies a simple DC gain and offset to the input data with a 0 output threshold:
        y = v1*x + v2
    where x is the input variable, and v1,v2 are fitted parameters. 
    """
    stack.append(ud.demo.simple_demo,thresh=True)
    
def advdemo00(stack):
    stack.append(ud.demo.adv_demo)
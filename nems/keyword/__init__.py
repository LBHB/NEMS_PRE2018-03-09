#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialization file for nems.keyword package

Created on Fri Aug 11 11:21:54 2017

@author: shofer
"""

import pkgutil as pk


#import nems.keyword.auxkeys 
import nems.keyword.est_valkeys
import nems.keyword.filterkeys 
import nems.keyword.fitterkeys 
import nems.keyword.loadkeys  
import nems.keyword.nonlinkeys 
import nems.keyword.pupilkeys
import nems.keyword.userkeys 


# build list of keyword functions that doesn't require knowing the sublibrary
# of the keyword (overload userkeys over defaults)
keyfuns={}
for importer, modname, ispkg in pk.iter_modules(__path__):
    f=importer.find_module(modname).load_module(modname)
    for k in dir(f):
        if callable(getattr(f,k)):
            keyfuns[k]=getattr(f,k)

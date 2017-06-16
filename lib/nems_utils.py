#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import scipy as sp
import numpy as np
import lib.nems_modules as nm

def find_modules(stack, mod_name):
    matchidx=[]
    for idx,m in enumerate(stack.modules):
        if m.name==mod_name:
            matchidx.append(idx)
 
    return matchidx


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import imp
import scipy.io
import pkgutil as pk

import nems.modules as nm
import nems.main as main
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities as nu
import nems.stack as ns
from nems.keyword.registry import keyword_registry

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

imp.reload(nm)
imp.reload(main)
imp.reload(nf)
imp.reload(nk)
imp.reload(nu)
imp.reload(ns)

site='TAR010c16'
factorN=3
#site='zee015h05'
doval=1
cellid="{0}-F{1}".format(site,factorN)

# load the stimulus
batch=271 #A1
modelname="fchan100_wc02_fir15_fit01"
stack=ns.nems_stack(cellid=cellid,batch=batch,modelname=modelname)
stack.meta['resp_channels']=[factorN]
stack.meta['site']=site
stack.keyfuns=0

stack.valmode=False

# evaluate the stack of keywords
if 'nested' in stack.keywords[-1]:
    # special case for nested keywords. Stick with this design?
    print('Using nested cross-validation, fitting will take longer!')
    k = stack.keywords[-1]
    keyword_registry[k](stack)
else:
    print('Using standard est/val conditions')
    for k in stack.keywords:
        print(k)
        keyword_registry[k](stack)

if doval:
    # validation stuff
    stack.valmode=True
    stack.evaluate(1)
    
    stack.append(nm.metrics.correlation)
    
    #print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
    #             stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
    valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx=valdata[0]
    else:
        stack.plot_dataidx=0

stack.quick_plot()

savefile = nu.io.get_file_name(cellid, batch, modelname)
nu.io.save_model(stack, savefile)



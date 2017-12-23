#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:14:38 2017

@author: svd
"""

import pandas.io.sql as psql
import matplotlib.pyplot as plt
import nems.utilities as nu
import nems.db as ndb
import nems.stack as ns
import nems.modules as nm
import nems.fitters as nf
import nems.keyword as nk
from nems.db import NarfResults, Session
import nems.poplib as poplib

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.decomposition import PCA
import copy
import imp

site='TAR010c16'
#site='TAR017b10'
#site='bbl086b09'
#site='zee015h05'
batch=271
fmodelname="fchan100_wcg02_fir15_fit01"
factorCount=5

outpath='/auto/users/svd/docs/current/grant/crcns_array/figures/raw/'

# fit factor STRFS:
for ii in range(0,factorCount):
    try:
        stack=poplib.factor_strf_load(site=site,factorN=ii,modelname=fmodelname)
        stack.quick_plot()
    except:
        stack=poplib.factor_strf_fit(site=site,factorN=ii,modelname=fmodelname)
    
    # save factor STRF plots:
    fig=plt.gcf()
    mode='pdf'
    filename = ("{0}{1}_{2}.{3}"
        .format(outpath, stack.meta['cellid'], fmodelname, mode)) 
    fig.savefig(filename)
    


stack=poplib.pop_factor_strf_init(site=site,factorCount=factorCount,batch=batch,fmodelname=fmodelname)
stack=poplib.pop_factor_strf_fit(stack)

poplib.pop_factor_strf_eval(stack, base_modelname="fb18ch100_wcg02_fir15_fit01")





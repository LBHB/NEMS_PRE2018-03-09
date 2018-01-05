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
from nems.keyword.registry import keyword_registry

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
fmodelname="fchan100_wc02_stp1pc_fir15_fit01"
factorCount=10

outpath='/auto/users/svd/docs/current/grant/crcns_array/figures/raw2/'

# fit factor STRFS:
for ii in range(0,factorCount):
    try:
        stack=poplib.factor_strf_load(site=site,factorN=ii,modelname=fmodelname)
        #stack.quick_plot()
    except:
        stack=poplib.factor_strf_fit(site=site,factorN=ii,modelname=fmodelname)
    
        # save factor STRF plots: only if re-calc-ed
        fig=plt.gcf()
        mode='pdf'
        filename = ("{0}{1}_{2}.{3}"
            .format(outpath, stack.meta['cellid'], fmodelname, mode)) 
        fig.savefig(filename)
    

plt.close('all')
factorCount=3

#modelname=fmodelname.replace("fchan100","ssfb18ch100")
#modelname=modelname.replace("_fit01","")
modelname="ssfb18ch100_wc02_stp1pc_fir15"

# fit the model
stack=poplib.pop_factor_strf_init(site=site,factorCount=factorCount,batch=batch,fmodelname=fmodelname,modelname=modelname)
stack=poplib.pop_factor_strf_fit(stack)

# or load a saved version
#stack=poplib.pop_factor_strf_load(site,factorCount,batch,modelname)

plt.close('all')

poplib.pop_factor_strf_eval(stack, base_modelname="fb18ch100_wc02_stp1pc_fir15_fit01")

mode='pdf'
fig=plt.figure(1)
filename = ("{0}{1}-F{2}_{3}_raster.{4}".format(outpath, stack.meta['site'], stack.meta['factorCount'], stack.meta['modelname'], mode)) 
fig.savefig(filename)
fig=plt.figure(2)
filename = ("{0}{1}-F{2}_{3}_pred.{4}".format(outpath, stack.meta['site'], stack.meta['factorCount'], stack.meta['modelname'], mode)) 
fig.savefig(filename)

stack.quick_plot()
fig=plt.gcf()
filename = ("{0}{1}-F{2}_{3}_strf.{4}".format(outpath, stack.meta['site'], stack.meta['factorCount'], stack.meta['modelname'], mode)) 
fig.savefig(filename)




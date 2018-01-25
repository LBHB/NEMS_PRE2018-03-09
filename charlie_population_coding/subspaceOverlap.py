#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:10:25 2018

@author: hellerc
"""

import os
import sys
sys.path.append('/auto/users/hellerc/nems/nems/utilities')
import imp
import baphy as bu
import nems.utilities as nu
from dimReduce_tools import PCA
import matplotlib.pyplot as plt
import nems.db as db
import numpy as np
import pandas as pd
import scipy.signal as ss
import charlie_random_utils as cru
# =================== Set parameters to find file =================
site='BOL006b'
runclass='VOC'  #(can't use until we migrate over the database)
batch = 294   # 289? - NAT/pup, 301 - PTD/pupil, 294 - VOC/pupil
rawid =  118758  # Ony need for PPS_VOC sets: BOL005c: 118702, BOL006b: 118758
pupil=1

iso=70
resample=0
samps=14
reduce_method='PCA';
# =================== parms of the cached files to load ======================    
parms={  
        'rasterfs':100,
        'runclass':'all',
        'min_isolation': iso,
        'includeprestim':1, 
        'tag_masks':['Reference']
    }
p_parms = {  
        'rasterfs':100,
        'runclass':'VOC',
        'min_isolation': iso,
        'includeprestim':1, 
        'pupil_median': 1,
        #'tag_masks':['Reference'],
        #'pupil_derivative': 'pos',  #pos or neg
        'pupil_lowpass': 0,  # 0 or n pass
        'pupil_highpass': 0 # 0 or n pass ex: 0.5

    }
# ========================= Load cellids from db =============================

r, meta = bu.load_site_raster(batch=batch,site=site,rawid=rawid,options=parms)
p = bu.load_pup_raster(batch=batch,site=site,options=p_parms,rawid=rawid)
try:
    p_parms['runclass']='VOC'
    p_parms['pupil_median']=0
    p_parms['pupil_lowpass']=0.05
    p_lp = bu.load_pup_raster(batch=batch,site=site,options=p_parms,rawid=rawid)
    p_parms['pupil_lowpass']=0
    p_parms['pupil_highpass']=0.05
    p_hp = bu.load_pup_raster(batch=batch,site=site,options=p_parms,rawid=rawid)
except:
   pass
try:
    #p_parms['pupil_median']=1
    p_parms['runclass']='VOC'
    p_parms['pupil_derivative']='pos'
    p_parms['pupil_highpass']=0
    p_dpos = bu.load_pup_raster(batch=batch,site=site,options=p_parms,rawid=rawid)
    p_parms['pupil_derivative']='neg'
    p_dneg = bu.load_pup_raster(batch=batch,site=site,options=p_parms,rawid=rawid)
except:
    pass

# ========================= Pre-process the data ============================
r, p = cru.remove_nans(runclass=runclass, options=parms, r=r, p=p)
try:
    p_lp = cru.remove_nans(runclass=runclass, options=parms, p=p_lp)
    p_hp = cru.remove_nans(runclass=runclass, options=parms, p=p_hp)
except:
    pass
try:
    p_dpos = cru.remove_nans(runclass=runclass, options=parms, p=p_dpos)
    p_dneg = cru.remove_nans(runclass=runclass, options=parms, p=p_dneg)
except:
    pass
# Resample if wanted    
if resample==1:
    r = ss.resample(r,samps)
    p = ss.resample(p,samps)
    
# Summarize dims for later use    
bincount=r.shape[0]
repcount=r.shape[1]
stimcount=r.shape[2]
cellcount=r.shape[3]

# Visualize pupil
plt.figure()
plt.title(site+'_'+runclass+'_'+'pupil')
bins = int(len(np.mean(p,0).flatten())/2);
out = plt.hist(np.mean(p,0).flatten(),bins=bins,color='green')

# --- TODO --- come up with a good way to divide into big/small
#count=out[0]
#pup_val=out[1]
#minima=pup_val[ss.argrelextrema(count,np.less,order=5)]
#plt.axvline(minima,color='k')

# ===================== Send data off for analysis ===========================

# call linear model function (regression between r and variabel set of predictors)
from regression_utils import linear_model
pred, rsq = linear_model(r, p)
r_no_p = (r.reshape(bincount*stimcount*repcount, cellcount)-pred).reshape(bincount, repcount, stimcount, cellcount)
#r = r_no_p

if reduce_method is 'PCA':
    pcs, var, step, loading = PCA(r,trial_averaged=False,center=True)
  


## ===========================================================================

'''   
# Filtering pupil...    
p_long = p.transpose(1,2,0).flatten()   
from scipy.signal import butter, lfilter, freqz
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

order = 6
fs = 100.0       # sample rate, Hz
cutoff = 1  # desired cutoff frequency of the filter, Hz 
   
y = butter_lowpass_filter(p_long, cutoff, fs, order)
plt.plot(y)
plt.plot(p_long)
'''


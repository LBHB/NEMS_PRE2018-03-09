#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:10:25 2018

@author: hellerc
"""

import os
import sys
sys.path.append('/auto/users/hellerc/nems/nems/utilities')
from baphy import load_spike_raster, spike_cache_filename, pupil_cache_filename, load_pupil_raster
import nems.utilities as nu
from dimReduce_tools import PCA
import matplotlib.pyplot as plt
import nems.db as db
import numpy as np
import pandas as pd
import scipy.signal as ss

# =================== Set parameters to find file =================
folder = '/auto/data/daq/Tartufo/TAR010/sorted/'
pfolder='/auto/data/daq/Tartufo/TAR010/'
site='TAR010c'
runclass='PTD'
runs = [9, 10, 11, 12]           # All runs and rawids must have been sorted together!
rawids = [123675, 123676, 123677, 123681]
batch = 301
pupil=1
iso=70
resample=1
samps=14
reduce_method='PCA';

# =================== parms of the cached files to load ======================    
parms={  
        'rasterfs':100,
        'runclass':'all',    
        'includeprestim':1,
        'tag_masks':['Reference'],
        'unit':1
    }
# ========================= Load cellids from db =============================
if len(runclass.split('_'))>1:
    for i, run in enumerate(runclass.split('_')):
        if i==0:
            d=db.get_cell_files(cellid=site,runclass=run)
            
        else:
            d = pd.concat([d,db.get_cell_files(cellid=site,runclass=run)])
else:
    d=db.get_cell_files(cellid=site,runclass=runclass)

cellids=np.sort(np.unique(d[d['rawid']==rawids[0]]['cellid']))    
    
isolation = []
for cellid in cellids:
    isolation.append(db.get_isolation(cellid, rawids[0])['isolation'].iloc[0])
isolation = np.array(isolation)

cellids=cellids[isolation>=iso]
ch_un = [unit[-4:] for unit in cellids]
cellcount=len(cellids)

pup_parms = parms
pup_parms['pupil']=pupil
# =============================================================================

# load data for all respfiles and pupfiles
a_p=[]
for i, cid in enumerate(cellids):
    d=db.get_batch_cell_data(batch,cid)
    respfile=nu.baphy.spike_cache_filename2(d['raster'],parms)
    pupfile=nu.baphy.pupil_cache_filename2(d['pupil'],pup_parms)
    for j, rf in enumerate(respfile):
        rts=nu.io.load_matlab_matrix(rf,key="r")
        pts=nu.io.load_matlab_matrix(pupfile.iloc[j],key='r')
        
        if '_a_' in rf:
            a_p = a_p+[1]*rts.shape[1]
        else:
            a_p = a_p+[0]*rts.shape[1]
        
        if j == 0:
            rt = rts;
            p = pts;
        else:
            rt = np.concatenate((rt,rts),axis=1)
            p = np.concatenate((p,pts),axis=1)
    if i == 0:
        r = np.empty((rt.shape[0],rt.shape[1],rt.shape[2],cellcount))
        r[:,:,:,0]=rt;
    else:
        r[:,:,:,i]=rt;
# ========================= Pre-process the data ============================

if runclass=='PPS_VOC':
    # last two stim are vocalizations, first 19 are pip sequences. This is for VOC
    prestim=2
    poststim=0.5
    duration=3
    r = r[0:int(parms['rasterfs']*(prestim+duration+poststim)),:,-2:,:];
    
elif runclass=='PTD':
    # Dropping any reps in which there were Nans for one or more stimuli (quick way
    # way to deal with nana. This should be improved)

    inds = []
    for ind in np.argwhere(np.isnan(r[0,:,:,0])):
        inds.append(ind[0])
    inds = np.array(inds)
    #drop_inds=np.unique(inds)
    keep_inds=[x for x in np.arange(0,len(r[0,:,0,0])) if x not in inds]
    
    a_p = np.array(a_p)[keep_inds]
    r = r[:,keep_inds,:,:]
    p = p[:,keep_inds,:]
   
    
elif runclass=='NAT':    
    # Different options for this... chop out the extra reps of first few? Or keep all data?
    # Chopping it out for now
    r = r[:,0:3,:,:]
    p = p[:,0:3,:]
    stim_inds = []
    
    for si in np.argwhere(np.isnan(r[0,0,:,0])):
        stim_inds.append(si[0])
    stim_inds=np.array(stim_inds)
    
    si_keep = [x for x in np.arange(0,len(r[0,0,:,0])) if x not in stim_inds]
    
    r = r[:,:,si_keep,:]
    p = p[:,:,si_keep]

    
# Resample if wanted    
if resample==1:
    r = ss.resample(r,samps)
    p = ss.resample(p,samps)
    
# Summarize dims for later use    
bincount=r.shape[0]
repcount=r.shape[1]
a_reps=sum(a_p)
p_reps=sum(a_p==0)
stimcount=r.shape[2]
cellcount=r.shape[3]
# ===================== Send data off for analysis ===========================

if reduce_method is 'PCA':
    pcs, var, step, loading = PCA(r,trial_averaged=False,center=True)
    pcs_a, var_a, step_a, loading_a = PCA(r[:,a_p==1,:,:],trial_averaged=False,center=True)
    pcs_p, var_p, step_p, loading_p = PCA(r[:,a_p==0,:,:],trial_averaged=False, center=True)
    

# call linear model function (regression between r and variabel set of predictors)

    
# =============================================================================    

cc_p=[]
cc_a=[]    
for i in range(0,cellcount):
     cc_p.append(np.corrcoef((pcs_p[:,i],p[:,a_p==0,:].reshape(bincount*p_reps*stimcount)))[0][1])
     cc_a.append(np.corrcoef((pcs_a[:,i],p[:,a_p==1,:].reshape(bincount*a_reps*stimcount)))[0][1])

# ===================== Figure 1 =======================================
fig1=1;

plt.figure(fig1)
plt.plot(var_a,'.-r')
plt.plot(var_p,'.-b')
plt.plot(var, '.-k',alpha=0.5)
plt.legend(['active','passive','both'])


## ===================== Figure 2 ========================================
fig2 = 2
npcs = cellcount
# plot the correlation btwn pupil and pcs
     
plt.figure(fig2)
plt.subplot(223)
plt.title('pupil correlations with pcs')
plt.bar(range(0,cellcount),cc_p,alpha=0.5, color='r')
plt.bar(range(0,cellcount),cc_a,alpha=0.5, color='b')
plt.legend(['passive','active'])
plt.xlabel('PCs')
plt.ylabel('r^2')

# plot the loading vectors    

cross_cor = np.matmul(loading_a[:,0:npcs],loading_p[:,0:npcs].T)    
vmin=np.min(np.hstack((loading_a,loading_p,cross_cor)))
vmax=np.max(np.hstack((loading_a,loading_p,cross_cor)))    
plt.figure(fig2)    
plt.subplot(221)
plt.title('Loading vector passive')
plt.imshow(loading_p[:,0:npcs],vmin=vmin,vmax=vmax,cmap='jet',aspect='auto')
plt.yticks(range(0,cellcount),cellids)
plt.xticks(range(0,npcs),range(0,npcs),rotation=90)
plt.xlabel('PCs')
plt.subplot(222)
plt.title('Loading vector active')    
plt.imshow(loading_a[:,0:npcs],vmin=vmin,vmax=vmax,cmap='jet',aspect='auto')
plt.yticks(range(0,cellcount),cellids)
plt.xlabel('PCs')
plt.xticks(range(0,npcs),range(0,npcs),rotation=90)

# plot correlation btwn loading vectors

plt.figure(fig2)
plt.subplot(224)
plt.title('Cross correlation')
plt.imshow(cross_cor,vmin=vmin,vmax=vmax,cmap='jet')
plt.yticks(range(0,cellcount),cellids)
plt.xticks(range(0,cellcount),cellids, rotation=90)
plt.colorbar()    
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
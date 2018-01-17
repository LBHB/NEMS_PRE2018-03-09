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
from dimReduce_tools import PCA
import matplotlib.pyplot as plt
import nems.db as db
import numpy as np
import pandas as pd
# =================== Set parameters to find file =================
folder = '/auto/data/daq/Tartufo/TAR010/sorted/'
pfolder='/auto/data/daq/Tartufo/TAR010/'
site='TAR010c'
runclass='PTD'
runs = [9, 10, 11, 12]           # All runs and rawids must have been sorted together!
rawids = [123675, 123676, 123677, 123681]
pupil=1
iso=74
# ============== Rim reduction method ============================
reduce_method='PCA';


# Load cellids from db
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

# =================== parms of the cached files to load ======================    
parms={  
        'rasterfs':100,
        'runclass':'all',    
        'includeprestim':'1',
        'tag_masks':['Reference'],
        'channel':ch_un[0][1],
        'unit':1
    }

pup_parms = parms
pup_parms['pupil']=pupil
# =============================================================================

# Get names of spk files to load
files=os.listdir(folder)

to_open = []
for f in files:
    for r in runs:
        if len(str(r))==1:
            r_str='0'+str(r)
        else:
            r_str=str(r)
            
        if runclass in f and site+r_str in f:
            to_open.append(f)
        else:
            pass
to_open=list(np.unique(to_open))
# get names of pup files to load
pfiles=os.listdir(pfolder)
p_files_to_open = []
for f in pfiles:
    for r in runs:
        if len(str(r))==1:
            r_str='0'+str(r)
        else:
            r_str=str(r)
        if runclass in f and 'pup.mat' in f and r_str in f:
            p_files_to_open.append(f)
        else:
            pass
p_files_to_open=list(np.unique(p_files_to_open))    
# ======================= Load the data =================================    
resp_list=[]
p_list=[]
act_pass=[]
for i, f in enumerate(to_open):
    
    if pup_parms['pupil']==1:
        ptemp = load_pupil_raster(pfolder+p_files_to_open[i],pup_parms)
        p_list.append(ptemp)
        
    samp = load_spike_raster(folder+f,parms)
    temp = np.empty((samp.shape[0],samp.shape[1],samp.shape[2],cellcount))
    
    if '_a_' in f:
        act_pass.append(np.repeat(1,samp.shape[1]))
    else:
        act_pass.append(np.repeat(0,samp.shape[1]))
    
    for i, cell in enumerate(ch_un):
        parms['unit']=cell[-1]
        if cell[0]=='0':
            parms['channel']=cell[1]
        else:
            parms['channel']=cell[0:2]
        
        fn=spike_cache_filename(f, parms)    
        rt, tags, trialset, exptevents = load_spike_raster(folder+f,parms,nargout=4)
        print(rt.shape)
        temp[:,:,:,i] = rt
        
    resp_list.append(temp)
    
r = np.concatenate(resp_list,axis=1)
a_p =np.concatenate(act_pass)

if pup_parms['pupil']==1:
    p = np.concatenate(p_list,axis=1)
    
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

# ===================== Send data off for analysis ===========================

if reduce_method is 'PCA':
    pcs, var, step, loading = PCA(r,center=True)

for i in range(0,cellcount):
     print(np.corrcoef((pcs[:,i],p.reshape(145*11*30)))[0][1])
    

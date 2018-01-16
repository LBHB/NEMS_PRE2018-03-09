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
import matplotlib.pyplot as plt
import nems.db as db
import numpy as np

# =================== Set parameters to load data from cache =================
folder = '/auto/data/daq/Tartufo/TAR010/sorted/'
pfolder='/auto/data/daq/Tartufo/TAR010/'
site='TAR010c'
runclass='NAT'
runs = [16]
pupil=1

# Load cellids from db
cache_path=folder+'cache/'
p_cache_path = pfolder+'tmp/'
cellids=np.unique(db.get_cell_files(cellid=site,runclass=runclass)['cellid'].values)
ch_un = [unit[-4:] for unit in cellids]
cellcount=len(cellids)

#parms of the cached files to load    
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
        rt = load_spike_raster(folder+f,parms)
        print(rt.shape)
        temp[:,:,:,i] = rt
        
    resp_list.append(temp)
    
r = np.concatenate(resp_list,axis=1)
a_p =np.concatenate(act_pass)

if pup_parms['pupil']==1:
    p = np.concatenate(p_list,axis=1)
    
# ========================= Pre-process the data ============================
    
    

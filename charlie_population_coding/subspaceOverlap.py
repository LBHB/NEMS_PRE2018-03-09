#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:10:25 2018

@author: hellerc
"""

from baphy import load_spike_raster, cache_filename
import matplotlib.pyplot as plt
import nems.db as db
import numpy as np
import os

folder = '/auto/data/daq/Tartufo/TAR010/sorted/'
site='TAR010c'
runclass='PTD'

files=os.listdir(folder)

to_open = []
for f in files:
    if 'PTD' in f:
        to_open.append(f)
    else:
        pass

cache_path=folder+'cache/'
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


resp_list=[]
act_pass=[]
for f in to_open:
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
        
        fn=cache_filename(f, parms)    
        rt = load_spike_raster(folder+f,parms)
        print(rt.shape)
        temp[:,:,:,i] = rt
    resp_list.append(temp)
    
r = np.concatenate(resp_list,axis=1)
a_p =np.concatenate(act_pass)
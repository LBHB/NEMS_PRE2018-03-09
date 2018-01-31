#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:40 2018

@author: svd
"""

import os
import io
import re
import numpy as np
import scipy.io
#import nems.recording as Recording
import pandas as pd
import matplotlib.pyplot as plt

import nems.utilities.baphy
import nems.signal

# figure out filepath for demo files
USE_LOCAL_DATA=False
if USE_LOCAL_DATA:
    nems_path=os.path.dirname(nems.utilities.__file__)
    t=nems_path.split('/')
    nems_root='/'.join(t[:-2]) + '/'
    nems.utilities.baphy.stim_cache_dir=nems_root+'signals/baphy_example/'
    nems.utilities.baphy.spk_subdir=''

# Behavior example
#cellid='BRT007c-a2'
#parmfilepath='/auto/data/daq/Beartooth/BRT007/BRT007c05_a_PTD.m'
#options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'parm', 'chancount': 0, 'cellid': cellid}
#event_times, spike_dict, stim_dict = nems.utilities.baphy.baphy_load_recording(parmfilepath,options)

# Nat sound + pupil example
cellid='TAR010c-CC-U'
if USE_LOCAL_DATA:
    parmfilepath=nems.utilities.baphy.stim_cache_dir+'TAR010c16_p_NAT.m'
else:
    parmfilepath='/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'
    options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}
    #parmfilepath='/auto/data/daq/Boleto/BOL005/BOL005c05_p_PPS_VOC.m'
    #options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf',
    #         'chancount': 18, 'cellid': 'all', 'pupil': True,'runclass': 'VOC'}

#cellid='TAR017b-CC-U'
#parmfilepath='/auto/data/daq/Tartufo/TAR017/TAR017b10_p_NAT.m'
#cellid='eno024d-b1'
#parmfilepath='/auto/data/daq/Enoki/eno024/eno024d10_p_NAT.m'
#pupilfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.pup.mat'

event_times, spike_dict, stim_dict, state_dict = nems.utilities.baphy.baphy_load_recording(parmfilepath,options)


# RDT example
#cellid="oys035b-a2"
#parmfilepath='/auto/data/daq/Oyster/oys035/oys035b04_p_RDT.m'
#options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 
#         'cellid': cellid, 'pertrial': True}
#event_times, spike_dict, stim_dict, stim1_dict, stim2_dict, state_dict = nems.utilities.baphy.baphy_load_recording_RDT(parmfilepath,options)


raster_all,cellids=nems.utilities.baphy.spike_time_to_raster(spike_dict,fs=options['rasterfs'],event_times=event_times)

resp=nems.signal.Signal(fs=options['rasterfs'],matrix=raster_all,name='resp',recording=cellid,chans=cellids,epochs=event_times)

r=resp.extract_epochs('00Oxford_male2b.wav')

# compute raster for specific unit and stimulus id with sampling rate rasterfs

unitidx=0 # which unit
eventidx=1

stimevents=list(stim_dict.keys())
cellids=list(spike_dict.keys())
cellids.sort()

event_name=stimevents[eventidx]
cellid=cellids[unitidx]

#event_name='TRIAL'

binlen=1.0/options['rasterfs']
h=np.array([])
ff = (event_times['name']==event_name)
## pull out each epoch from the spike times, generate a raster of spike rate
for i,d in event_times.loc[ff].iterrows():
    print("{0}-{1}".format(d['start'],d['end']))
    edges=np.arange(d['start'],d['end']+binlen,binlen)
    th,e=np.histogram(spike_dict[cellid],edges)
    
    print("{0}-{1}: {2}".format(edges[0],edges[1],sum((spike_dict[cellid]>edges[0]) & (spike_dict[cellid]<edges[1]))))
    th=np.reshape(th,[1,-1])
    if h.size==0:
        # lazy hack: intialize the raster matrix without knowing how many bins it will require
        h=th
    else:
        # concatenate this repetition, making sure binned length matches
        if th.shape[1]<h.shape[1]:
            h=np.concatenate((h,np.zeros([1,h.shape[1]])),axis=0)
            h[-1,:]=np.nan
            h[-1,:th.shape[1]]=th
        else:
            h=np.concatenate((h,th[:,:h.shape[1]]),axis=0)
    
m=np.nanmean(h,axis=0)

plt.figure()
plt.subplot(3,1,1)
plt.imshow(stim_dict[event_name],origin='lower',aspect='auto')
plt.title("stim {0}".format(event_name))
plt.subplot(3,1,2)
plt.imshow(h,origin='lower',aspect='auto')
plt.title("cell {0} raster".format(cellid))
plt.subplot(3,1,3)
plt.plot(np.arange(len(m))*binlen,m)
plt.title("cell {0} PSTH".format(cellid))
plt.tight_layout()


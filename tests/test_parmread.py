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
import nems.recording as Recording
import pandas as pd
import nems.utilities.baphy


# figure out filepath for demo files
nems_path=os.path.dirname(Recording.__file__)
t=nems_path.split('/')
nems_root='/'.join(t[:-1]) + '/'


# Behavior example
#cellid='BRT007c-a2'
#parmfilepath=nems_root+'signals/baphy_example/BRT007c05_a_PTD.m'
#spkfilepath=nems_root+'signals/baphy_example/BRT007c05_a_PTD.spk.mat'
#stimfilepath=nems_root+'signals/baphy_example/Torc2-0.35-0.35-8896-L125-4000_Hz__-0-0.75-55dB-parm-fs100-ch0-incps1.mat'
#options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'parm', 'chancount': 0, 'cellid': cellid}

# Nat sound + pupil example
cellid='TAR010c-CC-U'
parmfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.m'
options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all'}
#spkfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.spk.mat'
#stimfilepath=nems_root+'signals/baphy_example/NaturalSounds-2-0.5-3-1-White______-100-0-3__8-65dB-ozgf-fs100-ch18-incps1.mat'
#pupilfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.pup.mat'

# RDT example
#cellid="oys022b-b1"
#parmfilepath=nems_root+'signals/baphy_example/oys022c02_p_RDT.m'
#options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 
#         'cellid': cellid, 'pertrial': True}


# load the data
event_times, spike_dict, stim_dict = nems.utilities.baphy.baphy_load_recording(parmfilepath,options)


# compute raster for specific unit and stimulus id with sampling rate rasterfs
unitidx=0 # which unit
rasterfs=options['rasterfs']
binlen=1.0/rasterfs

stimevents=stim_dict.keys()

## pull out each epoch from the spike times, generate a raster of spike rate
#for i,d in eventtimes.iterrows():
#    print("{0}-{1}".format(d['StartTime'],d['StopTime']))
#    edges=np.arange(d['StartTime']-binlen/2,d['StopTime']+binlen/2,binlen)
#    th,e=np.histogram(spiketimes[unitidx],edges)
#    th=np.reshape(th,[1,-1])
#    if i==0:
#        # lazy hack: intialize the raster matrix without knowing how many bins it will require
#        h=th
#    else:
#        # concatenate this repetition, making sure binned length matches
#        if th.shape[1]<h.shape[1]:
#            h=np.concatenate((h,np.zeros([1,h.shape[1]])),axis=0)
#            h[-1,:]=np.nan
#            h[-1,:th.shape[1]]=th
#        else:
#            h=np.concatenate((h,th[:,:h.shape[1]]),axis=0)
#    
#m=np.nanmean(h,axis=0)
#
#import matplotlib.pyplot as plt
#plt.figure()
#plt.subplot(3,1,1)
#plt.imshow(stim[:,:,eventidx],origin='lower',aspect='auto')
#plt.title("stim {0} ({1})".format(eventidx, tags[eventidx]))
#plt.subplot(3,1,2)
#plt.imshow(h,origin='lower',aspect='auto')
#plt.title("cell {0} raster".format(unit_names[unitidx]))
#plt.subplot(3,1,3)
#plt.plot(np.arange(len(m))*binlen,m)
#plt.title("cell {0} PSTH".format(unit_names[unitidx]))
#plt.tight_layout()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:20:45 2018

@author: hellerc
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/auto/users/hellerc/NEMS/')
from nems.utilities.baphy import baphy_load_recording, spike_time_to_raster
from nems.signal import Signal
import pupil_processing as pp  

# load data in first (create a spike/response matrix)
rasterfs=100

# Tartufo NAT
#parmfilepath = '/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'
#options = {'rasterfs': rasterfs, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}

# Boleto VOC
parmfilepath = '/auto/data/daq/Boleto/BOL005/BOL005c05_p_PPS_VOC.m'
options = {'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True, 'runclass': 'VOC'}

# load baphy parmfile
out = baphy_load_recording(parmfilepath,
                           options)
# unpack output
event_times, spike_dict, stim_dict, state_dict = out
event_times['start_index']=[int(x) for x in event_times['StartTime']*100]
event_times['end_index']=[int(x) for x in event_times['StopTime']*100]

# r is response matrix, created from dictionary of spike times
out = spike_time_to_raster(spike_dict=spike_dict,
                         fs=100,
                         event_times=event_times)
r = out[0]
cellids = out[1]

# =========================================================================
# TODO
# Fiter out response matrix based on isolation... need db connection for this
# =========================================================================

# Create state matrix
p = state_dict['pupiltrace']
p_hp, p_lp = pp.filt(p, rasterfs=100)
dpos, dneg = pp.derivative(p, rasterfs=100)

state_matrix = np.vstack((p[:-1],p_hp[:-1],p_lp[:-1],dpos,dneg))

# Visualize pupil diamter
plt.figure()
n, bins, patches = plt.hist(p,bins=200)
l = plt.plot(bins, p, 'r--', linewidth=1)





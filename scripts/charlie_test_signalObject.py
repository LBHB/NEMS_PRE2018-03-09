#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 11:26:10 2018

@author: hellerc
"""
import sys
sys.path.append('/auto/users/hellerc/NEMS/')
import nems.utilities.baphy

# load data in first (matrix)

parmfilepath='/auto/data/daq/Tartufo/TAR010/TAR010c16_p_NAT.m'
options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 'chancount': 18, 'cellid': 'all', 'pupil': True}
event_times, spike_dict, stim_dict, state_dict = nems.utilities.baphy.baphy_load_recording(parmfilepath,options)

r = nems.utilities.baphy.spike_time_to_raster(spike_dict=spike_dict,fs=100,event_times=event_times)
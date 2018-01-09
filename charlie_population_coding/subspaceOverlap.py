#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:10:25 2018

@author: hellerc
"""

from baphy import load_spike_raster
import os

folder = '/auto/data/daq/Tartufo/TAR010/sorted/'

files=os.listdir(folder)

to_open = []
for f in files:
    if '.PTD.' == f:
        to_open.append(file)
    else:
        pass
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:10:25 2018

@author: hellerc
"""

from baphy import load_spike_raster, cache_filename
import nems.db as db
import numpy as np
import os

folder = '/auto/data/daq/Tartufo/TAR017/sorted/'
site='TAR017b'
runclass='PTD'

files=os.listdir(folder)

to_open = []
for f in files:
    if 'PTD' in f:
        to_open.append(f)
    else:
        pass

#parms of the cached files to load    
parms={  
        'rasterfs':100,
        'runclass':'all',    
        'includeprestim':'1',
        'tag_masks':['Reference'],
        'channel':3,
        'unit':1
    }

cache_path=folder+'cache/'
cellids=np.unique(db.get_cell_files(cellid=site,runclass=runclass)['cellid'].values)
ch_un = [unit[-4:] for unit in cellids]
for f in to_open:
    fn=cache_filename(f, parms)    
    print(fn)
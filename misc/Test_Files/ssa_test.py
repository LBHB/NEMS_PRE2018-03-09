#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:40:25 2017

@author: svd
"""
import baphy_utils
import pylab as pl
import numpy as np

f='/auto/data/code/nems_in_cache/batch296/eno035c-a1_b296_none_fs200.mat'

data=baphy_utils.load_baphy_file(f)

# spike matrix : Time X Rep X Stimulus Type
resp=data['resp']

# stimulus type labels
tags=data['tags'][0][0][3]


s=resp.shape

pl.figure()

#stimidx=0 # first noise frequency
stimidx=3 # second noise frequency

ax=pl.subplot(2,2,1);
d=np.transpose(resp[:,:,stimidx])
ax.imshow(d, aspect='auto', origin='lower')

ax=pl.subplot(2,2,2);
d=np.transpose(resp[:,:,stimidx+1])
ax.imshow(d, aspect='auto', origin='lower')

ax=pl.subplot(2,2,3);
d=resp[:,:,[stimidx, stimidx+1]]
psth=np.nanmean(d,1)
psth=np.squeeze(psth)
ax.plot(psth)




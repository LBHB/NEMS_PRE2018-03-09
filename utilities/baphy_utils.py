#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017

@author: svd
"""

import scipy.io as si
import numpy as np

def load_baphy_file(filepath):
    data=dict.fromkeys(['stim','resp','pup'])
    meta=dict.fromkeys(['stimf','respf','iso','prestim','duration','poststim',
                        'tags','tagidx','ff'])
    matdata=si.loadmat(filepath,chars_as_strings=True)
    m=matdata['data'][0][0]
    data['resp']=m['resp_raster']
    data['stim']=m['stim']
    meta['stimf']=m['stimfs'][0][0]
    meta['respf']=m['respfs'][0][0]
    meta['iso']=m['isolation'][0][0]
    meta['tags']=np.concatenate(m['tags'][0]['tags'][0][0],axis=0)
    meta['tagidx']=m['tags'][0]['tagidx'][0][0]
    meta['ff']=m['tags'][0]['ff'][0][0]
    meta['prestim']=m['tags'][0]['PreStimSilence'][0][0][0]
    meta['poststim']=m['tags'][0]['PostStimSilence'][0][0][0]
    meta['duration']=m['tags'][0]['Duration'][0][0][0] 
    try:
        data['pup']=m['pupil']
    except:
        data['pup']=None
    return(data,meta)
    
def get_celldb_file(batch,cellid,fs=200,stimfmt='ozgf',chancount=18):
    
    rootpath="/auto/data/code/nems_in_cache"
    fn="{0}/batch{1}/{2}_b{1}_{3}_c{4}_fs{4}".format(rootpath,batch,cellid,stimfmt,chancount,fs)
    
    # placeholder. Need to check if file exists in nems_in_cache.
    # If not, call baphy function in Matlab to regenerate it:
    # fn=export_cellfile(batchid,cellid,fs,stimfmt,chancount)
    
    return fn


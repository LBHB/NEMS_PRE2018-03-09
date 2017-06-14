#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017

@author: svd
"""

import scipy
import numpy as np

def load_baphy_file(f):

    matdata = scipy.io.loadmat(f,chars_as_strings=True)
    s=matdata['data'][0][0]
    data={} 
    data['resp']=s['resp_raster']   
    data['stim']=s['stim']
    data['respFs']=s['respfs']
    data['stimFs']=s['stimfs']
    data['stimparam']=[str(''.join(letter)) for letter in s['fn_param']]
    data['isolation']=s['isolation']
    data['tags']=s['tags']   
    
    return data

def get_celldb_file(batch,cellid,fs=200,stimfmt='ozgf',chancount=18):
    
    rootpath="/auto/data/code/nems_in_cache"
    fn="{0}/batch{1}/{2}_b{1}_{3}_c{4}_fs{4}".format(rootpath,batch,cellid,stimfmt,chancount,fs)
    
    # placeholder. Need to check if file exists in nems_in_cache.
    # If not, call baphy function in Matlab to regenerate it:
    # fn=export_cellfile(batchid,cellid,fs,stimfmt,chancount)
    
    return fn


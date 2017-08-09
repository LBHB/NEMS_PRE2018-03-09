#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017

@author: svd, changes added by njs 
"""

import scipy.io as si
#import numpy as np

try:
    import nems_config.Storage_Config as sc
except Exception as e:
    print(e)
    #import nems_config.STORAGE_DEFAULTS as sc

def load_baphy_file(filepath,level=0):
    """
    This function loads data from a BAPHY .mat file located at 'filepath'. 
    It returns two dictionaries contining the file data and metadata,
    'data' and 'meta', respectively. 'data' contains:
        {'stim':stimulus,'resp':response raster,'pupil':pupil diameters}
    Note that if there is no pupil data in the BAPHY file, 'pup' will return 
    None. 'meta' contains:
        {'stimf':stimulus frequency,'respf':response frequency,'iso':isolation,
             'tags':stimulus tags,'tagidx':tag idx, 'ff':frequency channel bins,
             'prestim':prestim silence,'duration':stimulus duration,'poststim':
             poststim silence}
    """
    data=dict.fromkeys(['stim','resp','pupil'])
    matdata=si.loadmat(filepath,chars_as_strings=True)
    s=matdata['data'][0][level]
    try:
        data={}
        data['resp']=s['resp_raster']
        data['stim']=s['stim']
        data['respFs']=s['respfs'][0][0]
        data['stimFs']=s['stimfs'][0][0]
        data['stimparam']=[str(''.join(letter)) for letter in s['fn_param']]
        data['isolation']=s['isolation']
        data['prestim']=s['tags'][0]['PreStimSilence'][0][0][0]
        data['poststim']=s['tags'][0]['PostStimSilence'][0][0][0]
        data['duration']=s['tags'][0]['Duration'][0][0][0] 
    except:
        data['raw_stim']=s['stim'].copy()
        data['raw_resp']=s['resp'].copy()
    try:
        data['pupil']=s['pupil']
    except:
        data['pupil']=None
    try:
        if s['estfile']:
            data['est']=True
        else:
            data['est']=False
    except ValueError:
        print("Est/val conditions not flagged in datafile")
    return(data)
    
def get_celldb_file(batch,cellid,fs=200,stimfmt='ozgf',chancount=18):
    """
    Given a stim/resp preprocessing parameters, figure out relevant cache filename.
    TODO: if cache file doesn't exist, have Matlab generate it
    
    @author: svd
    """
    
    rootpath=sc.DIRECTORY_ROOT + "nems_in_cache"
        
    fn="{0}/batch{1}/{2}_b{1}_{3}_c{4}_fs{5}.mat".format(rootpath,batch,cellid,stimfmt,chancount,fs)
    #fn='/auto/data/code/nems_in_cache/batch{0}/{1}_b{0}_{2}_c{3}_fs{4}.mat'.format(batch,cellid,stimfmt,chancount,fs)
    
    # placeholder. Need to check if file exists in nems_in_cache.
    # If not, call baphy function in Matlab to regenerate it:
    # fn=export_cellfile(batchid,cellid,fs,stimfmt,chancount)
    
    return fn


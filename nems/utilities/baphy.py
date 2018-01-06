#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017

@author: svd, changes added by njs
"""

import logging
log = logging.getLogger(__name__)

import os
import os.path
import scipy.io as si
import numpy as np

try:
    import nems_config.Storage_Config as sc
except Exception as e:
    log.info(e)
    from nems_config.defaults import STORAGE_DEFAULTS
    sc = STORAGE_DEFAULTS


def load_baphy_file(filepath, level=0):
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
    data = dict.fromkeys(['stim', 'resp', 'pupil'])
    matdata = si.loadmat(filepath, chars_as_strings=True)
    s = matdata['data'][0][level]
    print(s['fn_spike'])
    try:
        data = {}
        data['resp'] = s['resp_raster']
        data['stim'] = s['stim']
        data['respFs'] = s['respfs'][0][0]
        data['stimFs'] = s['stimfs'][0][0]
        data['stimparam'] = [str(''.join(letter)) for letter in s['fn_param']]
        data['isolation'] = s['isolation']
        data['prestim'] = s['tags'][0]['PreStimSilence'][0][0][0]
        data['poststim'] = s['tags'][0]['PostStimSilence'][0][0][0]
        data['duration'] = s['tags'][0]['Duration'][0][0][0]
        
        data['cellids']=s['cellids'][0]
        data['resp_fn']=s['fn_spike']
    except BaseException:
        data['raw_stim'] = s['stim'].copy()
        data['raw_resp'] = s['resp'].copy()
    try:
        data['pupil'] = s['pupil']
    except BaseException:
        data['pupil'] = None
    try:
        if s['estfile']:
            data['est'] = True
        else:
            data['est'] = False
    except ValueError:
        log.info("Est/val conditions not flagged in datafile")
    return(data)


def get_celldb_file(batch, cellid, fs=200, stimfmt='ozgf',
                    chancount=18, pertrial=False):
    """
    Given a stim/resp preprocessing parameters, figure out relevant cache filename.
    TODO: if cache file doesn't exist, have Matlab generate it

    @author: svd
    """

    rootpath = os.path.join(sc.DIRECTORY_ROOT, "nems_in_cache")
    if pertrial or batch in [269, 273, 284, 285]:
        ptstring = "_pertrial"
    else:
        ptstring = ""

    if stimfmt in ['none', 'parm', 'envelope']:
        fn = "{0}/batch{1}/{2}_b{1}{6}_{3}_fs{5}.mat".format(
            rootpath, batch, cellid, stimfmt, chancount, fs, ptstring)
    else:
        fn = "{0}/batch{1}/{2}_b{1}{6}_{3}_c{4}_fs{5}.mat".format(
            rootpath, batch, cellid, stimfmt, chancount, fs, ptstring)

    # placeholder. Need to check if file exists in nems_in_cache.
    # If not, call baphy function in Matlab to regenerate it:
    # fn=export_cellfile(batchid,cellid,fs,stimfmt,chancount)

    return fn


def load_baphy_ssa(filepath):
    """
    Load SSAs from matlab, cherrypiking the most convenient values
    of the mat file and parsing them into a dictionary. The mat file contains
    a struct wich can be a vector (multiple related recording of the same cell)
    therefore each poin in this vector is parced into a dictionary within a list.
    """
    matdata = si.loadmat(filepath, chars_as_strings=True)
    d = matdata['data']
    datalist = list()
    for i in range(d.size):
        m = d[0][i]
        data = dict.fromkeys(['stim', 'resp', 'tags'])
        try:
            params = m['stimparam'][0][0]
            data['PipDuration'] = round(params['Ref_PipDuration'][0][0], 4)
            data['PipInterval'] = round(params['Ref_PipInterval'][0][0], 4)
            Freq = params['Ref_Frequencies'].squeeze()
            data['Frequencies'] = Freq.tolist()
            Rates = params['Ref_F1Rates'].squeeze()
            data['Rates'] = Rates.tolist()
            data['Jitter'] = params['Ref_Jitter'][0][:]
            data['MinInterval'] = round(params['Ref_MinInterval'][0][0], 4)
        except BaseException:
            pass

        data['stimfmt'] = m['stimfmt'][0]

        if m['stimfmt'][0] == 'envelope':
            resp = np.swapaxes(m['resp_raster'], 0, 2)
            stim = np.squeeze(m['stim'])  # stim envelope seems to not be binay
            stim = stim / stim.max()
            stim = np.where(stim < 0.5, 0, 1)  # trasnforms stim to binary
            stim = np.swapaxes(stim, 1, 2)
            stim = stim[:, :, 0:resp.shape[2]]

            data['stim'] = stim

        elif m['stimfmt'][0] == 'none':
            data['stim'] = []
            resp = np.swapaxes(m['resp_raster'], 0, 1)

        data['resp'] = resp

        data['stimf'] = m['stimfs'][0][0]
        respf = m['respfs'][0][0]
        data['respf'] = respf
        data['isolation'] = round(m['isolation'][0][0], 4)
        try:
            data['tags'] = np.concatenate(m['tags'][0]['tags'][0][0], axis=0)
        except BaseException:
            pass
        try:
            data['tagidx'] = m['tags'][0]['tagidx'][0][0]
            data['ff'] = m['tags'][0]['ff'][0][0]
        except BaseException:
            pass
        prestim = m['tags'][0]['PreStimSilence'][0][0][0]
        data['prestim'] = prestim
        duration = m['tags'][0]['Duration'][0][0][0]
        data['duration'] = duration
        poststim = resp.shape[2] - (int(prestim) + int(duration)) * int(respf)
        data['poststim'] = poststim / respf

        try:
            data['pup'] = m['pupil']
        except BaseException:
            data['pup'] = None
        datalist.append(data)

    return (datalist)

def load_spike_raster(spkfile, options):
    '''
    # CRH added 1-5-2018, work in progress - meant to mirror baphy's loadspikeraster
    
    inputs:
        spkfile - name of .spk.mat file generated using meska

        options - structure can contain the following fields:
            channel - electrode channel (default 1)
            unit - unit number (default 1)
            rasterfs in Hz (default 1000)
            includeprestim - raster includes silent period before stimulus onset
            tag_masks - cell array of strings to filter tags, eg,
                {'torc','reference'} or {'target'}.  AND logic.  default ={}
            psthonly - shape of r (default -1, see below)
            sorter - preferentially load spikes sorted by sorter.  if not
                sorted by sorter, just take primary sorting
            lfpclean - [0] if 1, remove MSE prediction of spikes (single
                trial) predicted by LFP
            includeincorrect - if 1, load all trials, not just correct (default 0)
            runclass - if set, load only data for runclass when multiple runclasses
                in file
     outputs:
         TODO define    
    '''
    
    
    # ========= This needs to be beefed up to work like loadspikeraster======
    # parse the input in options
    try: channel =options['channel']
    except: channel=1
    
    try: unit=options['unit']
    except: unit=1
    
    try: rasterfs=options['rasterfs']
    except: rasterfs=1000.
    
    try: tag_masks=options['tag_masks']; tag_name=tag_masks[0];
    except: tag_masks=[]; tag_name='Reference';
    
    
    # ========== see if cache file exists =====================
    need_matlab=0 # if set to 1, use matlab engine to generate the cache file
    
    # get path to spkfile
    if(len(spkfile.split('/'))==1):
        path_to_spkfile = os.getcwd()   
    else: 
        path_to_spkfile = os.path.dirname(spkfile)
    
    
    # define the cache file name
    spkfile_root_name=os.path.basename(spkfile).split('.')[0];
    cache_fn=spkfile_root_name+'_ch'+str(channel)+'-'+str(unit)+'_fs'+str(int(rasterfs))+'_'+tag_name+'.mat'
    
    
    # make cache directory if it doesn't already exist
    path_to_cacheFile = os.path.join(path_to_spkfile,'cache')
    cache_file = os.path.join(path_to_cacheFile,cache_fn)
    print(cache_file)
    if(os.path.isdir(path_to_cacheFile) and os.path.exists(cache_file)):
        out = si.loadmat(cache_file)
        
    elif(os.path.isdir(path_to_cacheFile)):
        need_matlab=1        
    else:
        need_matlab = 1
        os.mkdir(path_to_cacheFile)
    

    # Generate the cache file so that it can be loaded by python
    if need_matlab:
        # only start the matlab engine if the cached file doesn't exist
        import matlab.engine
        eng = matlab.engine.start_matlab()
        baphy_util_path = '/auto/users/hellerc/baphy/Utilities'
        eng.addpath(baphy_util_path,nargout=0)
        # call matlab function to make the requested array
        eng.loadspikeraster(spkfile, options, nargout=0)    #TODO figure out a way to specify nargout without returning anything
        
        out = si.loadmat(cache_file)
        
    return out
    
    
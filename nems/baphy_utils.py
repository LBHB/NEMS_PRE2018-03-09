#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 09:33:47 2017

@author: svd, changes added by njs 
"""

import scipy.io as si
import numpy as np

def load_baphy_file(filepath):
    """
    This function loads data from a BAPHY .mat file located at 'filepath'. 
    It returns two dictionaries contining the file data and metadata,
    'data' and 'meta', respectively. 'data' contains:
        {'stim':stimulus,'resp':response raster,'pup':pupil diameters}
    Note that if there is no pupil data in the BAPHY file, 'pup' will return 
    None. 'meta' contains:
        {'stimf':stimulus frequency,'respf':response frequency,'iso':isolation,
             'tags':stimulus tags,'tagidx':tag idx, 'ff':frequency channel bins,
             'prestim':prestim silence,'duration':stimulus duration,'poststim':
             poststim silence}
    """
    data=dict.fromkeys(['stim','resp','pup'])
    meta=dict.fromkeys(['stimf','respf','iso','prestim','duration','poststim',
                        'tags','tagidx','ff'])
    matdata=si.loadmat(filepath,chars_as_strings=True)
    m=matdata['data'][0][0]
    data['resp']=m['resp_raster']
    data['stim']=m['stim']
    data['stimf']=m['stimfs'][0][0]
    data['respf']=m['respfs'][0][0]
    data['iso']=m['isolation'][0][0]
    data['tags']=np.concatenate(m['tags'][0]['tags'][0][0],axis=0)
    try:
        data['tagidx']=m['tags'][0]['tagidx'][0][0]
        data['ff']=m['tags'][0]['ff'][0][0]
    except:
        pass    
    data['prestim']=m['tags'][0]['PreStimSilence'][0][0][0]
    data['poststim']=m['tags'][0]['PostStimSilence'][0][0][0]
    data['duration']=m['tags'][0]['Duration'][0][0][0] 
    try:
        data['pup']=m['pupil']
    except:
        data['pup']=None
    return(data)
    
def get_celldb_file(batch,cellid,fs=200,stimfmt='ozgf',chancount=18):
    """
    Given a stim/resp preprocessing parameters, figure out relevant cache filename.
    TODO: if cache file doesn't exist, have Matlab generate it
    
    @author: svd
    """
    rootpath="/auto/data/code/nems_in_cache"
    fn="{0}/batch{1}/{2}_b{1}_{3}_c{4}_fs{5}.mat".format(rootpath,batch,cellid,stimfmt,chancount,fs)
    
    # placeholder. Need to check if file exists in nems_in_cache.
    # If not, call baphy function in Matlab to regenerate it:
    # fn=export_cellfile(batchid,cellid,fs,stimfmt,chancount)
    
    return fn


def get_kw_file(batch,cellid,keyword):
    """
    Given a keyword, translate to stim/resp preprocessing parameters and get relevant filename
    
    @author: svd
    """
       
    lookup={};
    
    lookup['fb18ch100']={}
    lookup['fb18ch100']['fs']=100
    lookup['fb18ch100']['stimfmt']='ozgf'
    lookup['fb18ch100']['chancount']=18
    lookup['fb18ch200']={}
    lookup['fb18ch200']['fs']=200
    lookup['fb18ch200']['stimfmt']='ozgf'
    lookup['fb18ch200']['chancount']=18
    lookup['fb18ch400']={}
    lookup['fb18ch400']['fs']=400
    lookup['fb18ch400']['stimfmt']='ozgf'
    lookup['fb18ch400']['chancount']=18
    lookup['fb24ch100']={}
    lookup['fb24ch100']['fs']=100
    lookup['fb24ch100']['stimfmt']='ozgf'
    lookup['fb24ch100']['chancount']=24
    lookup['fb36ch100']={}
    lookup['fb36ch100']['fs']=100
    lookup['fb36ch100']['stimfmt']='ozgf'
    lookup['fb36ch100']['chancount']=36
    lookup['fb48ch100']={}
    lookup['fb48ch100']['fs']=100
    lookup['fb48ch100']['stimfmt']='ozgf'
    lookup['fb48ch100']['chancount']=36
    lookup['env100']={}
    lookup['env100']['fs']=100
    lookup['env100']['stimfmt']='envelope'
    lookup['env100']['chancount']=0
    lookup['env200']={}
    lookup['env200']['fs']=200
    lookup['env200']['stimfmt']='envelope'
    lookup['env200']['chancount']=0
   
    if keyword == '' or keyword==None:
        fn='no preproc, just use filename passed by NEMS_analysis'
        
    elif keyword in lookup.keys():
        fs=lookup[keyword]['fs']
        stimfmt=lookup[keyword]['stimfmt']
        chancount=lookup[keyword]['chancount']
        fn=get_celldb_file(batch,cellid,fs,stimfmt,chancount)
        
    else:
        raise NameError('keyword not found')
    
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
        data = dict.fromkeys(['stim','resp','tags'])
        try:
            params = m['stimparam'][0][0]
            data['PipDuration'] = round(params['Ref_PipDuration'][0][0],4)
            data['PipInterval'] = round(params['Ref_PipInterval'][0][0],4)
            data['MinInterval'] = round(params['Ref_MinInterval'][0][0],4)
            data['Jitter'] = params['Ref_Jitter'][0][:]
            Rates = params['Ref_F1Rates'].squeeze()
            data['Rates'] = Rates.tolist()
            Freq = params['Ref_Frequencies'].squeeze()
            data['Frequencies'] = Freq.tolist()
        except:
            pass

        resp = np.swapaxes(m['resp_raster'],0,2)

        data['resp'] = resp

        data['stimfmt'] = m['stimfmt'][0]

        if m['stimfmt'][0] == 'envelope':
            stim = np.squeeze(m['stim'])  # stim envelope seems to not be binay
            stim = stim / stim.max()
            stim = np.where(stim < 0.5, 0, 1)  # trasnforms stim to binary
            stim = np.swapaxes(stim, 1, 2)
            stim = stim[:,:,0:resp.shape[2]]

            data['stim'] = stim

        elif m['stimfmt'][0] == 'none':
            data['stim'] = []


        data['stimf'] = m['stimfs'][0][0]
        respf = m['respfs'][0][0]
        data['respf'] = respf
        data['isolation'] = round(m['isolation'][0][0],4)
        data['tags'] = np.concatenate(m['tags'][0]['tags'][0][0], axis=0)
        try:
            data['tagidx'] = m['tags'][0]['tagidx'][0][0]
            data['ff'] = m['tags'][0]['ff'][0][0]
        except:
            pass
        prestim = m['tags'][0]['PreStimSilence'][0][0][0]
        data['prestim'] = prestim
        duration = m['tags'][0]['Duration'][0][0][0]
        data['duration'] = duration
        poststim = resp.shape[2] - (prestim + duration) * respf
        data['poststim'] = poststim / respf

        try:
            data['pup'] = m['pupil']
        except:
            data['pup'] = None
        datalist.append(data)

    return (datalist)
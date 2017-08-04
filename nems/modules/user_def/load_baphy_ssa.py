#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:25:25 2017

@author: shofer
"""
import numpy as np
import scipy.io as si
import nems.utilities.utils as nu
from nems.modules.base import nems_module


class load_baphy_ssa(nems_module):
    """
    Load SSAs from matlab, cherrypiking the most convenient values
    of the mat file and parsing them into a dictionary. The mat file contains
    a struct wich can be a vector (multiple related recording of the same cell)
    therefore each poin in this vector is parced into a dictionary within a list.
    
    @author: mateo
    """
    def my_init(self,file=[],fs=1000):
        self.filepath=file
        self.parent_stack.avg_resp=False
        self.fs=fs

    def evaluate(self,**kwargs):
        del self.d_out[:]
        filepath=self.filepath
        matdata = si.loadmat(filepath, chars_as_strings=True)
        d = matdata['data']
        #datalist = []
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
            data['resp'] = m['resp_raster']
    
            data['stimfmt'] = m['stimfmt'][0]
  
    
            data['stimFs'] = m['stimfs'][0][0]
            data['respFs'] = m['respfs'][0][0]
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
            poststim = data['resp'].shape[0] - (prestim + duration) * data['respFs']
            data['poststim'] = -poststim / data['respFs']

            print(data['prestim'],data['duration'],data['poststim'])
    
            try:
                data['pupil'] = m['pupil']
            except:
                data['pupil'] = None
            
            sr=data['resp'].shape[0]/data['respFs']
            data['duration']=sr-data['prestim']
            
            
            self.parent_stack.unresampled={'resp':data['resp'],'respFs':data['respFs'],'duration':data['duration'],
                                               'pupil':data['pupil']}
            
            if m['stimfmt'][0] == 'envelope':
                stim = np.squeeze(m['stim'])  # stim envelope seems to not be binary
                stim = stim / stim.max()
                stim = np.where(stim < 0.5, 0, 1)  # trasnforms stim to binary
                stim = np.swapaxes(stim, 1, 2)
                stim = stim[:,:,0:data['resp'].shape[0]]
                data['poststim']=0
                self.parent_stack.unresampled['prestim']=data['prestim']
                self.parent_stack.unresampled['poststim']=data['poststim']
                data['stim'] = stim
                
            elif m['stimfmt'][0] == 'none':
                data['stim'] = []
            
            
            #Resample data, if desired:    
            data['fs']=self.fs
            noise_thresh=0.04
            stim_resamp_factor=int(data['stimFs']/self.fs)
            resp_resamp_factor=int(data['respFs']/self.fs)
            
            if stim_resamp_factor != 1:
                data['stim']=nu.thresh_resamp(data['stim'],stim_resamp_factor,thresh=noise_thresh,ax=2)
                
            if resp_resamp_factor != 1:
                data['resp']=nu.thresh_resamp(data['resp'],resp_resamp_factor,thresh=noise_thresh)
                
            if data['pupil'] is not None and resp_resamp_factor != 1:
                data['pupil']=nu.thresh_resamp(data['pupil'],resp_resamp_factor,thresh=noise_thresh)
                
            data['repcount']=np.sum(np.isnan(data['resp'][0,:,:])==False,axis=0)
            self.parent_stack.unresampled['repcount']=data['repcount']
            #datalist.append(data)
            data['stim'],data['resp'],data['pupil'],data['replist']=nu.stretch_trials(data)
    
        self.d_out.append(data)
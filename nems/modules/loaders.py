#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules for importing data into the stack object


Created on Fri Aug  4 13:14:24 2017

@author: shofer
"""
from nems.modules.base import nems_module
import numpy as np
import nems.utilities.utils as nu
import scipy.io

class load_mat(nems_module):
    """
    Loads a MATLAB data file (.mat file) containing several "structs" which have 
    data for an individual cell. 
    
    Inputs:
        fs: frequency to resample stimulus, response, and pupil data. 
        avg_resp: average all trials in the response raster and place in 
                the output dictionary as 'resp'. Usually used when pupil
                effect are being considered, and will generally allow for
                better fitting.
        est_files: MATLAB data files to load. 
    
    Returns: Data from this file is loaded into the stack.data as a list of dictionaries 
    with keywords:
        'resp': response raster for each type of stimulus
        'stim': stimuli spectrograms that correspond to response
        'respFs': sampling frequency of response raster
        'stimFs':sampling frequency of stimuli spectrograms
        'stimparam': details on types of stimuli used
        'isolation': isolation of recorded cells (?)
        'prestim': length of silence before stimulus begins
        'poststim': length of silence after stimulus ends
        'duration': length of simulus
        'pupil': continuous pupil diameter measurements
        'est': flag for estimation/validation data
        'repcount': how many trials of each stimulus are present
        'replist': a list containing the number of each stimulus the number of
                times it was played. E.g., if we have stimulus 1 that was played
                3 times and stimulus 2 that was played 2 times, replist would
                be [1,1,1,2,2].
    """
    name='loaders.load_mat'
    user_editable_fields=['output_name','est_files','fs']
    plot_fns=[nu.plot_spectrogram, nu.plot_spectrogram]
    est_files=[]
    fs=100
    
    def my_init(self,est_files=[],fs=100,avg_resp=True):
        self.est_files=est_files.copy()
        self.fs=fs
        self.avg_resp=avg_resp
        self.parent_stack.avg_resp=avg_resp
        self.auto_plot=False
        self.save_dict={'est_files':self.est_files,'fs':fs,'avg_resp':avg_resp}

    def evaluate(self,**kwargs):
        del self.d_out[:]
#        for i, d in enumerate(self.d_in):
#            self.d_out.append(d.copy())
                    
        # load contents of Matlab data file
        for f in self.est_files:
            matdata = nu.get_mat_file(f)
            for s in matdata['data'][0]:
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
                    data = scipy.io.loadmat(f,chars_as_strings=True)
                    data['raw_stim']=data['stim'].copy()
                    data['raw_resp']=data['resp'].copy()
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
                    
                                    
                data['fs']=self.fs
                noise_thresh=0.05
                stim_resamp_factor=int(data['stimFs']/self.fs)
                resp_resamp_factor=int(data['respFs']/self.fs)
                
                self.parent_stack.unresampled={'resp':data['resp'],'respFs':data['respFs'],'duration':data['duration'],
                                               'poststim':data['poststim'],'prestim':data['prestim'],'pupil':data['pupil']}
                
                # reshape stimulus to be channel X time
                data['stim']=np.transpose(data['stim'],(0,2,1))
                if stim_resamp_factor != 1:
                    data['stim']=nu.thresh_resamp(data['stim'],stim_resamp_factor,thresh=noise_thresh,ax=2)
                    
                # resp time (axis 0) should be resampled to match stim time (axis 1)
                if resp_resamp_factor != 1:
                    data['resp']=nu.thresh_resamp(data['resp'],resp_resamp_factor,thresh=noise_thresh)
                    
                if data['pupil'] is not None and resp_resamp_factor != 1:
                    data['pupil']=nu.thresh_resamp(data['pupil'],resp_resamp_factor,thresh=noise_thresh)
                    
                #Changed resample to decimate w/ 'fir' and threshold, as it produces less ringing when downsampling
                #-njs June 16, 2017
                    
                # average across trials
                data['repcount']=np.sum(np.isnan(data['resp'][0,:,:])==False,axis=0)
                self.parent_stack.unresampled['repcount']=data['repcount']
                
                data['avgresp']=np.nanmean(data['resp'],axis=1)
                data['avgresp']=np.transpose(data['avgresp'],(1,0))

                if self.avg_resp is True: 
                    data['resp']=data['avgresp']
                else:
                    data['stim'],data['resp'],data['pupil'],data['replist']=nu.stretch_trials(data)

                # append contents of file to data, assuming data is a dictionary
                # with entries stim, resp, etc...
                #print('load_mat: appending {0} to d_out stack'.format(f))
                self.d_out.append(data)
                
class dummy_data(nems_module):
    """
    dummy_data - generate some very dumb test data without loading any files. 
    Maybe deprecated? 
    """
    name='loaders.dummy_data'
    user_editable_fields=['output_name','data_len']
    plot_fns=[nu.plot_spectrogram]
    data_len=100
    
    def my_init(self,data_len=100):
        self.data_len=data_len
        self.save_dict={'data_len':data_len}

    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
        
        self.d_out[0][self.output_name]=np.zeros([12,2,self.data_len])
        self.d_out[0][self.output_name][0,0,10:19]=1
        self.d_out[0][self.output_name][0,0,30:49]=1
        self.d_out[0]['resp']=self.d_out[0]['stim'][0,:,:]*2+1        
        self.d_out[0]['repcount']=np.sum(np.isnan(self.d_out[0]['resp'])==False,axis=0)
        
        

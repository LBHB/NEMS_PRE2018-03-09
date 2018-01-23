#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:28:43 2018

@author: hellerc
"""
import numpy as np


def remove_nans(runclass, options, r, p=None):
    '''
    Specific to this script. Remove nans based on given runclass. Return r and p
    or just r, depending on if pupil is specified
    '''
    parms = options
    if p is not None:
    
        if runclass=='VOC':
            # last two stim are vocalizations, first 19 are pip sequences. This is for VOC
            prestim=2
            poststim=0.5
            duration=3
            r = r[0:int(parms['rasterfs']*(prestim+duration+poststim)),:,-2:,:];
            p = p[0:int(parms['rasterfs']*(prestim+duration+poststim)),:,-2:];
            
        elif runclass=='PTD':
            # Dropping any reps in which there were Nans for one or more stimuli (quick way
            # way to deal with nana. This should be improved)
        
            inds = []
            for ind in np.argwhere(np.isnan(r[0,:,:,0])):
                inds.append(ind[0])
            inds = np.array(inds)
            #drop_inds=np.unique(inds)
            keep_inds=[x for x in np.arange(0,len(r[0,:,0,0])) if x not in inds]
            
            r = r[:,keep_inds,:,:]
            p = p[:,keep_inds,:]
           
            
        elif runclass=='NAT':    
            # Different options for this... chop out the extra reps of first few? Or keep all data?
            # Chopping it out for now
            r = r[:,0:3,:,:]
            p = p[:,0:3,:]
            stim_inds = []
            
            for si in np.argwhere(np.isnan(r[0,0,:,0])):
                stim_inds.append(si[0])
            stim_inds=np.array(stim_inds)
            
            si_keep = [x for x in np.arange(0,len(r[0,0,:,0])) if x not in stim_inds]
            
            r = r[:,:,si_keep,:]
            p = p[:,:,si_keep]
            
        return r, p
            
    else:
        
        if runclass=='VOC':
            # last two stim are vocalizations, first 19 are pip sequences. This is for VOC
            prestim=2
            poststim=0.5
            duration=3
            r = r[0:int(parms['rasterfs']*(prestim+duration+poststim)),:,-2:,:];
            
        elif runclass=='PTD':
            # Dropping any reps in which there were Nans for one or more stimuli (quick way
            # way to deal with nana. This should be improved)
        
            inds = []
            for ind in np.argwhere(np.isnan(r[0,:,:,0])):
                inds.append(ind[0])
            inds = np.array(inds)
            #drop_inds=np.unique(inds)
            keep_inds=[x for x in np.arange(0,len(r[0,:,0,0])) if x not in inds]
            
            r = r[:,keep_inds,:,:]
           
            
        elif runclass=='NAT':    
            # Different options for this... chop out the extra reps of first few? Or keep all data?
            # Chopping it out for now
            r = r[:,0:3,:,:]
            stim_inds = []
            
            for si in np.argwhere(np.isnan(r[0,0,:,0])):
                stim_inds.append(si[0])
            stim_inds=np.array(stim_inds)
            
            si_keep = [x for x in np.arange(0,len(r[0,0,:,0])) if x not in stim_inds]
            
            r = r[:,:,si_keep,:]
            
        return r
        

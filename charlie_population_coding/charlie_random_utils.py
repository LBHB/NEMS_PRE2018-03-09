#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 17:28:43 2018

@author: hellerc
"""
import numpy as np




def create_raster(spike_dict, fs, tend):
    '''
    Given spike times, sampling rate, and length of recording, create a spike 
    raster, (time x neurons)
    
    CRH, 01-27-2018
    '''    
    cellids=[x for x in spike_dict.keys()]
    ncells=len(cellids)
    exptbins=int(tend*fs)
    r=np.zeros((exptbins,ncells))
    
    for cell, cid in enumerate(cellids):
        spike_times=spike_dict[cid]
        exptbins=int(tend*fs)
        spikebins=np.array([int(x) for x in fs*spike_times])
        spikebins=spikebins[spikebins<exptbins]
        for i in spikebins:
            r[i, cell]+=1

    return r
    
    
def create_event_raster(spike_dict, event_times, fs, stimulus=None):
    '''
    Given spike times, event times, sampling rate, length of recording
    produce a raster sorted by trial stimulus and repetition. If stimulus is 
    specified, return a raster of tbins x stim x rep x neuron
    '''
    if stimulus is None:
        stimuli=len(event_times['epoch_name'].unique())-3
        trials=int(event_times['epoch_name'].value_counts().iloc[3])  # get first that isn't trial, pre, or post stim
        bins=int(round(np.max(event_times['StopTime']-event_times['StartTime']),2)*fs)
        ncells=len([x for x in spike_dict.keys()])
        
        r = np.zeros((bins,stimuli,trials,ncells))
        
        stims=event_times['epoch_name'].unique()
        stims = [e for e in stims if e not in {'PreStimSilence', 'TRIAL','PostStimSilence'}]
        
        cellids=np.sort(np.array([x for x in spike_dict.keys()]))
        
        for c, cid in enumerate(cellids):
            print('loading raster for: ' + cid)
            for s, stim in enumerate(stims):
                reps = sum(event_times['epoch_name']==stim)
                events_this_stim=event_times[event_times['epoch_name']==stim]
                for rep in range(0,reps):
                    start=int(events_this_stim['StartTime'].loc[rep]*fs)
                    end=int(events_this_stim['StopTime'].loc[rep]*fs)
                    spikes=[int(x) for x in (spike_dict[cid]*fs) if x > start and x < end]
                    spikes=np.array(spikes)-int(events_this_stim['StartTime'].loc[rep]*fs)
                    for t in spikes:
                        r[t,s,rep,c]+=1
        return r
    else:
        stimuli=1
        trials=int(event_times[event_times['epoch_name']==stimulus]['epoch_name'].value_counts().iloc[0])  # get first that isn't trial, pre, or post stim
        bins=int(round(np.max(event_times['StopTime']-event_times['StartTime']),2)*fs)
        ncells=len([x for x in spike_dict.keys()])
        
        r = np.zeros((bins,stimuli,trials,ncells))
        
        stims=[stimulus]
        
        cellids=np.sort(np.array([x for x in spike_dict.keys()]))
        
        for c, cid in enumerate(cellids):
            print('loading raster for: ' + cid)
            for s, stim in enumerate(stims):
                reps = sum(event_times['epoch_name']==stim)
                events_this_stim=event_times[event_times['epoch_name']==stim]
                for rep in range(0,reps):
                    start=int(events_this_stim['StartTime'].loc[rep]*fs)
                    end=int(events_this_stim['StopTime'].loc[rep]*fs)
                    spikes=[int(x) for x in (spike_dict[cid]*fs) if x > start and x < end]
                    spikes=np.array(spikes)-int(events_this_stim['StartTime'].loc[rep]*fs)
                    for t in spikes:
                        r[t,s,rep,c]+=1
                        
        
        return r


def remove_nans(runclass, options, r=None, p=None):
    '''
    Specific to this script. Remove nans based on given runclass. Return r and p
    or just r, depending on if pupil is specified
    '''
    parms = options
    if p is not None and r is not None:
    
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
            
    elif r is not None and p is None:
        
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
    
    elif p is not None and r is None:
        
        if runclass=='VOC':
            # last two stim are vocalizations, first 19 are pip sequences. This is for VOC
            prestim=2
            poststim=0.5
            duration=3
            p = p[0:int(parms['rasterfs']*(prestim+duration+poststim)),:,-2:];
            
        elif runclass=='PTD':
            # Dropping any reps in which there were Nans for one or more stimuli (quick way
            # way to deal with nana. This should be improved)
        
            inds = []
            for ind in np.argwhere(np.isnan(r[0,:,:])):
                inds.append(ind[0])
            inds = np.array(inds)
            #drop_inds=np.unique(inds)
            keep_inds=[x for x in np.arange(0,len(p[0,:,0])) if x not in inds]
            
            p = p[:,keep_inds,:]
           
            
        elif runclass=='NAT':    
            # Different options for this... chop out the extra reps of first few? Or keep all data?
            # Chopping it out for now
            p = p[:,0:3,:]
            stim_inds = []
            
            for si in np.argwhere(np.isnan(p[0,0,:])):
                stim_inds.append(si[0])
            stim_inds=np.array(stim_inds)
            
            si_keep = [x for x in np.arange(0,len(p[0,0,:])) if x not in stim_inds]
            
            p = p[:,:,si_keep]
            
        return p
        

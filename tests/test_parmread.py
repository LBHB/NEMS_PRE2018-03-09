#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:40 2018

@author: svd
"""

import os
import io
import re
import numpy as np
import scipy.io
import nems.recording as Recording
import pandas as pd


def baphy_mat2py(s):
    
    s3=re.sub(r';', r'', s.rstrip())
    s3=re.sub(r'%',r'#',s3)
    s3=re.sub(r'\\',r'/',s3)
    s3=re.sub(r"\.([a-zA-Z0-9]+)'",r"XX\g<1>'",s3)
    s3=re.sub(r"\.([a-zA-Z0-9]+) ,",r"XX\g<1> ,",s3)
    s3=re.sub(r'globalparams\(1\)',r'globalparams',s3)
    s3=re.sub(r'exptparams\(1\)',r'exptparams',s3)
              
    s4=re.sub(r'\(([0-9]*)\)', r'[\g<1>]', s3)
#    s4=re.sub(r"\.(?m')(?evp')", r"XXXX", s4)
#    s4=re.sub(r"\.(?m')(?evp')", r"XXXX", s4)
    s5=re.sub(r'\.([A-Za-z][A-Za-z1-9_]+)', r"['\g<1>']", s4)
    
    s6=re.sub(r'([0-9]+) ', r"\g<0>,", s5)
    s6=re.sub(r'NaN ', r"np.nan,", s6)
    
    s7=re.sub(r"XX([a-zA-Z0-9]+)'",r".\g<1>'",s6)
    s7=re.sub(r"XX([a-zA-Z0-9]+) ,",r".\g<1> ,",s7)
    #s7=re.sub(r"XXevp'",r".evp'",s7)
    
    s7=re.sub(r',,',r',',s7)
    s7=re.sub(r'NaN',r'np.nan',s7)
    s7=re.sub(r'zeros\(([0-9,]+)\)',r'np.zeros([\g<1>])',s7)
    s7=re.sub(r'{(.*)}',r'[\g<1>]',s7)
    
    return s7

def baphy_parm_read(filepath):
    
    f = io.open(filepath, "r")
    
    s=f.readlines(-1)
    
    globalparams={}
    exptparams={}
    exptevents={}
    
    for ts in s:
        sout=baphy_mat2py(ts)
        #print(sout)
        try:
            exec(sout)
        except KeyError:
            ts1=sout.split('= [')
            ts1=ts1[0].split(',[')
            
            s1=ts1[0].split('[')
            sout1="[".join(s1[:-1]) + ' = {}'
            try: 
                exec(sout1)
            except :
                s2=sout1.split('[')
                sout2="[".join(s2[:-1]) + ' = {}'
                exec(sout2)
                exec(sout1)
            exec(sout)
        except NameError:
            print("NameError on: {0}".format(sout))
        except:
            print("Other error on: {0} to {1}".format(ts,sout))

    # special conversions
    
    # convert exptevents to a DataFrame:
    t=[exptevents[k] for k in exptevents]
    d=pd.DataFrame(t)
    exptevents=d.drop(['Rove'],axis=1)
    for i in range(0,len(exptevents)):
        if exptevents.loc[i,'StopTime'] == []:
            exptevents.loc[i,'StopTime']=exptevents.loc[i,'StartTime']
    
    return globalparams, exptparams, exptevents

def baphy_load_specgram(stimfilepath):
    
    matdata = scipy.io.loadmat(stimfilepath, chars_as_strings=True)
    
    # remove redundant tags from tag list and stimulus array
    d=matdata['stimparam'][0][0][0][0]
    d=[x[0] for x in d]
    tags,tagids=np.unique(d, return_index=True)
    
    stim=matdata['stim']
    stim=stim[:,:,tagids]
    
    return stim,tags


def baphy_load_spike_data_raw(spkfilepath,channel=None,unit=None):
    
    matdata = scipy.io.loadmat(spkfilepath, chars_as_strings=True)

    sortinfo=matdata['sortinfo'][0]
    
    # figure out sampling rate, used to convert spike times into seconds
    spikefs=matdata['rate'][0][0]
        
    return sortinfo,spikefs

def baphy_align_time(exptevents,sortinfo,spikefs):
    
    # number of channels in recording (not all necessarily contain spikes)
    chancount=len(sortinfo)
    
    # figure out how long each trial is by the time of the last spike count.
    # this method is a hack! but since recordings are longer than the "official"
    # trial end time reported by baphy, this method preserves extra spikes
    TrialLen_spikefs=np.zeros([TrialCount+1,1])
    
    for c in range(0,chancount):
        if sortinfo[c].size:
            s=sortinfo[c][0][0]['unitSpikes']
            s=np.reshape(s,(-1,1))
            unitcount=s.shape[0]
            for u in range(0,unitcount):
                st=s[u,0]
                
                print('chan {0} unit {1}: {2} spikes'.format(c,u,st.shape[1]))
                for trialidx in range(1,TrialCount+1):
                    ff=(st[0,:]==trialidx)
                    if np.sum(ff):
                        utrial_spikefs=np.max(st[1,ff])
                        TrialLen_spikefs[trialidx,0]=np.max([utrial_spikefs,TrialLen_spikefs[trialidx,0]])
    
    # using the trial lengths, figure out adjustments to trial event times. 
    Offset_spikefs=np.cumsum(TrialLen_spikefs)
    Offset_sec=Offset_spikefs / spikefs  # how much to offset each trial
    
    # adjust times in exptevents to approximate time since experiment started
    # rather than time since trial started (native format)
    for Trialidx in range(1,TrialCount+1):
        print("Adjusting trial {0} by {1} sec".format(Trialidx,Offset_sec[Trialidx-1]))
        ff= (exptevents['Trial'] == Trialidx)
        exptevents.loc[ff,['StartTime','StopTime']]=exptevents.loc[ff,['StartTime','StopTime']]+Offset_sec[Trialidx-1]
    
    
    # convert spike times from samples since trial started to
    # (approximate) seconds since experiment started (matched to exptevents)
    totalunits=0
    spiketimes=[]  # list of spike event times for each unit in this recording
    unit_names=[]  # string suffix for each unit (CC-U)
    for c in range(0,chancount):
        if sortinfo[c].size:
            s=sortinfo[c][0][0]['unitSpikes']
            s=np.reshape(s,(-1,1))
            unitcount=s.shape[0]
            for u in range(0,unitcount):
                st=s[u,0]
                uniquetrials=np.unique(st[0,:])
                print('chan {0} unit {1}: {2} spikes {3} trials'.format(c,u,st.shape[1],len(uniquetrials)))
                
                unit_spike_events=np.array([])
                for trialidx in uniquetrials:
                    ff=(st[0,:]==trialidx)
                    this_spike_events=st[1,ff]+Offset_spikefs[np.int(trialidx-1)]
                    unit_spike_events=np.concatenate((unit_spike_events,this_spike_events),axis=0)
                
                totalunits+=1
                unit_names.append("{0:02d}-{1}".format(c+1,u+1))
                spiketimes.append(unit_spike_events / spikefs)
    
    return exptevents,spiketimes,unit_names    



# figure out filepath for demo files
nems_path=os.path.dirname(Recording.__file__)
t=nems_path.split('/')
nems_root='/'.join(t[:-1]) + '/'

# Behavior example
cellid='BRT007c-a1'
parmfilepath=nems_root+'signals/baphy_example/BRT007c05_a_PTD.m'
spkfilepath=nems_root+'signals/baphy_example/BRT007c05_a_PTD.spk.mat'
stimfilepath=nems_root+'signals/baphy_example/Torc2-0.35-0.35-8896-L125-4000_Hz__-0-0.75-55dB-parm-fs100-ch0-incps1.mat'

# Nat sound + pupil example
#cellid='TAR010c-CC-U'
#parmfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.m'
#spkfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.spk.mat'
#stimfilepath=nems_root+'signals/baphy_example/NaturalSounds-2-0.5-3-1-White______-100-0-3__8-65dB-ozgf-fs100-ch18-incps1.mat'
#pupilfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.pup.mat'


# load parameter file
globalparams, exptparams, exptevents = baphy_parm_read(parmfilepath)

TrialCount=np.max(exptevents['Trial'])


# load stimulus spectrogram
stim,tags = baphy_load_specgram(stimfilepath)

# load spike times
sortinfo,spikefs=baphy_load_spike_data_raw(spkfilepath)

# adjust spike and event times to be in seconds since experiment started
exptevents,spiketimes,unit_names = baphy_align_time(exptevents,sortinfo,spikefs)


# compute raster for specific unit and stimulus id with sampling rate rasterfs
unitidx=5  # which unit
unitidx=0  # which unit
eventidx=0
rasterfs=100.0


tag_mask_start="PreStimSilence , "+tags[eventidx]+" , Reference"
tag_mask_stop="PostStimSilence , "+tags[eventidx]+" , Reference"
binlen=1.0/rasterfs

ffstart=(exptevents['Note'] == tag_mask_start)
ffstop=(exptevents['Note'] == tag_mask_stop)

eventtimes=pd.concat([exptevents.loc[ffstart,['StartTime']].reset_index(), 
                      exptevents.loc[ffstop,['StopTime']].reset_index()], axis=1)

for i,d in eventtimes.iterrows():
    print("{0}-{1}".format(d['StartTime'],d['StopTime']))
    edges=np.arange(d['StartTime']-binlen/2,d['StopTime']+binlen/2,binlen)
    th,e=np.histogram(spiketimes[unitidx],edges)
    th=np.reshape(th,[1,-1])
    if i==0:
        # lazy hack: intialize the raster matrix without knowing how many bins it will require
        h=th
    else:
        # concatenate this repetition, making sure binned length matches
        h=np.concatenate((h,th[:,:h.shape[1]]),axis=0)
    
m=np.mean(h,axis=0)

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(3,1,1)
plt.imshow(stim[:,:,eventidx],origin='lower',aspect='auto')
plt.title("stim {0} ({1})".format(eventidx, tags[eventidx]))
plt.subplot(3,1,2)
plt.imshow(h,origin='lower',aspect='auto')
plt.title("cell {0} raster".format(unit_names[unitidx]))
plt.subplot(3,1,3)
plt.plot(np.arange(len(m))*binlen,m)
plt.title("cell {0} PSTH".format(unit_names[unitidx]))
plt.tight_layout()


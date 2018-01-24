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
    
    return globalparams, exptparams, exptevents


def baphy_load_spike_data(spkfilepath,channel=None,unit=None):
    
    #matdata = nems.utilities.io.get_mat_file(spkfilepath)
    matdata = scipy.io.loadmat(spkfilepath, chars_as_strings=True)
    
    


cellid='BRT007c-a1'
channel=1
unit=1

# figure out filepath for demo files
nems_path=os.path.dirname(Recording.__file__)
t=nems_path.split('/')
nems_root='/'.join(t[:-1]) + '/'

# Behavior example
#parmfilepath=nems_root+'signals/baphy_example/BRT007c05_a_PTD.m'
#spkfilepath=nems_root+'signals/baphy_example/BRT007c05_a_PTD.spk.mat'
#stimfilepath=nems_root+'signals/baphy_example/Torc2-0.35-0.35-8896-L125-4000_Hz__-0-0.75-55dB-parm-fs100-ch0-incps1.mat'

# Nat sound + pupil example
parmfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.m'
spkfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.spk.mat'
stimfilepath=nems_root+'signals/baphy_example/NaturalSounds-2-0.5-3-1-White______-100-0-3__8-65dB-ozgf-fs100-ch18-incps1.mat'
pupilfilepath=nems_root+'signals/baphy_example/TAR010c16_p_NAT.pup.mat'


# load parameter file
globalparams, exptparams, exptevents = baphy_parm_read(parmfilepath)

TrialCount=np.max(exptevents['Trial'])


# load spike times
matdata = scipy.io.loadmat(spkfilepath, chars_as_strings=True)

spikefs=matdata['rate'][0][0]
chancount=len(matdata['sortinfo'][0])
TrialLen_spikefs=np.zeros([TrialCount+1,1])

for c in range(0,chancount):
    if matdata['sortinfo'][0][c].size:
        s=matdata['sortinfo'][0][c][0][0]['unitSpikes']
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
                
Offset_spikefs=np.cumsum(TrialLen_spikefs)
Offset_sec=Offset_spikefs / spikefs

for Trialidx in range(1,TrialCount+1):
    
    ff=exptevents['Trial'] == Trialidx
    exptevents.loc[ff,['StartTime','StopTime']]=exptevents.loc[ff,['StartTime','StopTime']]+Offset_sec[Trialidx-1,0]


# convert spike times to absolute seconds
totalunits=0
spiketimes=[]
unit_names=[]
for c in range(0,chancount):
    if matdata['sortinfo'][0][c].size:
        s=matdata['sortinfo'][0][c][0][0]['unitSpikes']
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
            unit_names.append("{0}-{1}".format(c,u))
            spiketimes.append(unit_spike_events / spikefs)




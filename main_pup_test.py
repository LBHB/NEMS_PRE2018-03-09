#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:02:27 2017

@author: shofer
"""
import numpy as np
import nems.keywords as nk
import nems.utils as nu
import nems.baphy_utils as bu
import nems.modules as nm
import nems.stack as ns
import nems.fitters as nf
import nems.main as mn
import os
import os.path
import copy
"""
filelist=os.listdir('/auto/users/shofer/data/batch294')
files=[]
for i in filelist:
    f=i.split('_')
    files.append(f[0])
"""
#stack.meta['batch']=294
#stack.meta['cellid']='eno022e-b1'
"""modlist=['nopupgain','pupgain','polypupgain02','polypupgain03','polypupgain04','exppupgain','logpupgain',
         'butterworth01','butterworth02','butterworth03','butterworth04','poissonpupgain']"""
#BOL006b-43-1
#stack=mn.fit_single_model('BOL006b-43-1', 293, 'fb16ch50u_wc03_fir10_powergain02_fit02', autoplot=True)
#alldata=stack.data
#stack=mn.fit_single_model('eno052d-a1', 294, 'perfectpupil50_nopupgain_fit01', autoplot=True,crossval=False)
#print(slist.__len__())
#dat1=slist[1].data
#dat2=slist[19].data[-1][1]['repcount'].shape[0]

#valarr=np.split(slist[0].data[-1][1]['stim'],slist[0].data[-1][1]['repcount'].shape[0],0)
#for i in slist[1:]:
    #valarr=np.append(valarr,i.data[-1][1]['stim'],axis=0)
    
cellid='eno052b-c1'
batch=293
modelname="parm50_wc03_fir10_dexp_fit02"

stack=mn.fit_single_model(cellid, batch, modelname)    
    
#alldata=copy.deepcopy(stack.data[0])
#allmods=copy.deepcopy(stack.modules)

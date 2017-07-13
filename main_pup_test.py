#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:02:27 2017

@author: shofer
"""

import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.baphy_utils as bu
import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_main as mn
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

stack=mn.fit_single_model('eno052d-a1', 294, 'perfectpupil100_powergain02_fit01', autoplot=True,crossval=True)

alldata=copy.deepcopy(stack.data)


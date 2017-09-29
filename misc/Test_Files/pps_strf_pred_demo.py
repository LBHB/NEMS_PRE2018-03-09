#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import imp
import scipy.io
import pkgutil as pk

import nems.modules as nm
import nems.main as main
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities as ut
import nems.stack as ns

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

imp.reload(nm)
imp.reload(main)
imp.reload(nf)
imp.reload(nk)
imp.reload(ut)
imp.reload(ns)

"""
mysql> select cellid from NarfBatches WHERE cellid like "BOL005%" and batch=293 order by cellid;
+--------------+
| cellid       |
+--------------+
| BOL005c-04-1 |
| BOL005c-06-1 |
| BOL005c-07-1 |
| BOL005c-09-1 |
| BOL005c-09-3 |
| BOL005c-10-1 |
| BOL005c-13-1 |
| BOL005c-13-2 |
| BOL005c-16-3 |
| BOL005c-18-1 |
| BOL005c-22-2 |
| BOL005c-25-1 |
| BOL005c-25-5 |
| BOL005c-27-1 |
| BOL005c-29-1 |
| BOL005c-33-1 |
| BOL005c-34-1 |
| BOL005c-37-1 |
| BOL005c-40-1 |
| BOL005c-43-1 |
| BOL005c-44-1 |
| BOL005c-44-2 |
| BOL005c-46-1 |
| BOL005c-47-1 |
| BOL005c-48-1 |
| BOL005c-49-1 |
| BOL005c-52-1 |
+--------------+
"""


cellid='BOL005c-18-1'
batch=293
modelname="parm50_wcg02_fir10_pupgainctl_fit01_nested10"

stack=ut.io.load_single_model(cellid, batch, modelname)

fir_module_idx=ut.utils.find_modules(stack,'filters.fir')
val_file_idx=1   #entry in data stack that contains validation set

m=stack.modules[fir_module_idx[0]]

pred=m.d_out[val_file_idx]['stim'].copy()
resp=m.d_out[val_file_idx]['resp'].copy()
pupil=m.d_out[val_file_idx]['pupil'].copy()

m.do_plot(m)

r_val=stack.meta['r_val']  # test set prediction accuracy


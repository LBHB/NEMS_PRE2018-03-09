#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:18:15 2017

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
import nems.user_def_mods.load_baphy_ssa as lbs



batch=296
cellid='gus019d-a1'
keyword='jitterload_fir15_fit00'

stack=mn.fit_single_model(cellid,batch,keyword,autoplot=True)

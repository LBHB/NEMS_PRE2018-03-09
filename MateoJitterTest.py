#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 18:18:15 2017

@author: shofer
"""

import nems.main as mn




batch=296
cellid='gus018d-d1'
keyword='jitterload_fir15_dexp_fit02'

stack=mn.fit_single_model(cellid,batch,keyword,autoplot=True)

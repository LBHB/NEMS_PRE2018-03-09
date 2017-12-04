#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 16:20:48 2017

@author: jacob
"""

# testing Model / Signal changes

import nems.Model as mdl
import nems.Signal as sig
import nems.modules as mods
import nems.fitters.fitters as fit

modules = [mods.filters.weight_channels,
           mods.filters.fir,
           mods.nonlin.gain]

x = sig.load_signal('/home/ivar/sigs/gus027b13_p_PPS_pupil')
y = sig.load_signal('/home/ivar/sigs/gus027b15_p_PPS_pupil')

m = mdl.Model(modules=modules,
              fitter=fit.basic_min)

result1 = m.fit(x)
result2 = m.fit(y)

# use Signal module to combine x and y
# xy = TODO
# result3 = m.fit(xy)

print('result 1:')
print(result1)
print('result 2:')
print(result2)

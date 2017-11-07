#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:14:38 2017

@author: svd
"""


import pandas.io.sql as psql
from nems.db import NarfResults, Session
import matplotlib.pyplot as plt


batch=301
modelnames=['parm100pt_wcg02_fir15_pupgainctl_fit01_nested5',
           'parm100pt_wcg02_fir15_pupgain_fit01_nested5',
           'parm100pt_wcg02_fir15_stategain_fit01_nested5'
           ]


results = psql.read_sql_query(
     session.query(NarfResults)
     .filter(NarfResults.batch == batch)
     .filter(NarfResults.modelname.in_(modelnames))
     .statement,
     session.bind
     )

results.head()

mapping = {modelname: i for i, modelname in enumerate(modelnames)}
results['modelindex'] = results.modelname.map(mapping)

results.set_index(['modelindex', 'cellid'])['r_test'].unstack('modelindex').T.plot().legend(bbox_to_anchor=(1, 1))


r_test = results.set_index(['modelindex', 'cellid'])['r_test'].unstack('modelindex')

#plt.plot(r_test)

plt.figure()
f, ax = plt.subplots(1, 1)
ax.plot(r_test.T)

m=r_test.mean()
e=r_test.sem()

ax.

ax.xaxis.set_ticklabels(modelnames)
ax.xaxis.set_ticks([0, 1, 2])



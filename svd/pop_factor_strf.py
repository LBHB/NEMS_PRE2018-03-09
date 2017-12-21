#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:14:38 2017

@author: svd
"""

import pandas.io.sql as psql
import matplotlib.pyplot as plt
import nems.utilities as nu
import nems.db as ndb
import nems.stack as ns
import nems.modules as nm
import nems.fitters as nf
import nems.keyword as nk
from nems.db import NarfResults, Session
import nems.poplib as poplib

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
from sklearn.decomposition import PCA
import copy


site='TAR010c16'
#site='zee015h05'
doval=1
batch=271
modelname="fchan100_wc02_fir15_fit01"


stack=poplib.factor_strf_fit(site=site,factorN=0)
stack=poplib.factor_strf_fit(site=site,factorN=1)
stack=poplib.factor_strf_fit(site=site,factorN=2)
stack=poplib.factor_strf_fit(site=site,factorN=3)

stack=poplib.pop_factor_strf(site=site)




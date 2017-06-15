#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:24:23 2017

@author: shofer
"""

from nemsclass import FERReT
import plots_pack.raster_plot as rp
import plots_pack.coeff_heatmap as ch
import plots_pack.comparison as pc
import numpy as np
import utilities.baphy_utils as ubu

"""
This is designed to be a brief tutorial for several features of NEMS, including data
importation, plotting, fitting a model, and saving model parameters and predictions
"""
#Importing data:
pupilfile='/auto/users/shofer/data/batch294/eno048f-b1_nat_export.mat'
data,meta=ubu.load_baphy_file(filepath=pupilfile)

#Plotting a raster:
rp.raster_plot(stims='all',size=(12,6),data=data['resp'],pre_time=meta['prestim'],
               post_time=meta['poststim'],dur_time=meta['duration'],frequency=meta['respf'])

pupilmodel=FERReT(filepath=pupilfile,imp_type='standard',newHz=50,keyword='dlog_fir15_pupgain')



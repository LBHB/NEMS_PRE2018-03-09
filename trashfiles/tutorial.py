#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 18:24:23 2017

@author: shofer
"""
import numpy as np
from nemsclass import FERReT
import plots_pack.raster_plot as rp
import plots_pack.coeff_heatmap as ch
import plots_pack.comparison as pc
#import utilities.baphy_utils as ubu

"""
This is designed to be a brief tutorial for several features of NEMS, including data
importation, plotting, fitting a model, and saving model parameters and predictions
"""
"""
Importing data:
   To start, let's import a dataset that contains continuous pupil data"""
    
pupilfile='/auto/users/shofer/data/batch294/eno048f-b1_nat_export.mat'
#data,meta=ubu.load_baphy_file(filepath=pupilfile)
#help(ubu.load_baphy_file) #Call help() to see details about the function

"""Plotting a raster:"""
rp.raster_plot(stims='all',size=(12,6),data=data['resp'],pre_time=meta['prestim'],
               post_time=meta['poststim'],dur_time=meta['duration'],frequency=meta['respf'])

"""
Instantiate an example of the FERReT class:

    Note that since we specified a filepath the FERReT object, it will actually import
    the data for us. However, we could also have just passed it the data we imported earlier
    """
model=FERReT(filepath=pupilfile,imp_type='standard',newHz=50,keyword='dlog_fir15_pupgain_dexp')

"""
Now let's fit a model! Currently, we can tell the fitter to just make a validation and
training dataset for us. This will change in the future, but for now, since we only have 
two stimuli in this dataset, let's choose  validation size of 0.5. We will have the fitter
run 2 successive fits to generate our model. This may take some time (~15 minutes) on a
slow computer. Note that after the first evaluation, the "returns" of fitting decrease 
quickly. We will then save the fitted parameters to a file so that we do not need to
re-fit the model if we kill our evaluation kernel. (Also, you should probably change the 
filepath, unless you have the same directories as me)
"""
#model.run_fit(validation=0.5,reps=2,save=True,
              #filepath='/auto/users/shofer/Saved_Data/tutorial_params.npy')

"""Now, apply our fitted parameters to the validation and training data to generate predictions!"""

#valdata=model.apply_to_val(save=True,filepath='/auto/users/shofer/Saved_Data/tutorial_vals.npy')
#traindata=model.apply_to_train(save=True,filepath='/auto/users/shofer/Saved_Data/tutorial_train.npy')

"""
If we want to make changes to this tutorial and run it again without fitting again, we just 
need to load the saved data and comment out the fitter and apply_to functions:
"""
valdata=np.load('/auto/users/shofer/Saved_Data/tutorial_vals.npy')[()]
traindata=np.load('/auto/users/shofer/Saved_Data/tutorial_train.npy')[()]

pc.pred_vs_resp.count=0
pc.pred_vs_resp(traindata,obj=model,stims='all',trials=(10,15),size=(12,4))
pc.pred_vs_resp(valdata,obj=model,stims='all',trials=(10,15),size=(12,4))









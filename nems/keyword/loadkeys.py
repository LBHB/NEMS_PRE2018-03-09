#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loader keywords

Created on Fri Aug 11 10:34:40 2017

@author: shofer
"""

import nems.modules as nm
import nems.utilities as ut

def parm100(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def parm50(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def parm50a(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Averages the response 
    to each stimulus over its respective raster, and applies a 5% 
    estimation/validation split if the est/val datasets are not specified in 
    the file.
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)

def fb24ch200(stack):
    """
    Loads a 24 channel, 200 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=200,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def fb24ch100(stack):
    """
    Loads a 24 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True) #Data not preprocessed to 100 Hz, internally converts
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def fb18ch100(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
    
def fb18ch100u(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Does not average over response rasters, instead treating each trial as a separate
    stimulus. Applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.est_val.crossval,valfrac=stack.valfrac)
      
def fb18ch50(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch,
    then downsamples to 50 Hz.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=True)
    #stack.append(nm.crossval,valfrac=stack.valfrac)
    stack.append(nm.est_val.standard_est_val,valfrac=stack.valfrac)

def loadlocal(stack):
    """
    This keyword is just to load up a local file that is not yet on the BAPHY database.
    Right now just loads files from my computer --njs, June 27 2017
    """
    file='/Users/HAL-9000/Desktop/CompNeuro/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'_b'+str(stack.meta['batch'])+'_ozgf_c18_fs100.mat'
    #file=ut.baphy_utils.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval)
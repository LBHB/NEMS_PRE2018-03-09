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
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.est_val.crossval)
    
def env50e(stack):
    """
    Loads a 50 Hz BAPHY .mat file with 'envelope' marker using the 
    provided cellid and batch. Then compute and replace stim with envelope onsets.
    
    Specifically for batch296 SSA data
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='envelope')
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=0.00)
    stack.append(nm.aux.onset_edges)
    
def env100e(stack):
    """
    Loads a 100 Hz BAPHY .mat file with 'envelope' marker using the 
    provided cellid and batch. Then compute and replace stim with envelope onsets.
    
    Specifically for batch296 SSA data
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='envelope')
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=0.1)
    stack.append(nm.aux.onset_edges)


def env100ej(stack):
    """
    Loads a 100 Hz BAPHY .mat file with 'envelope' marker using the
    provided cellid and batch. Then compute and replace stim with envelope onsets.

    only loads index of the mat file where the field 'filestate' is 1

    Specifically for batch296 SSA data, in this case 'filestate' == 1 means Jitter On
    """
    file = ut.baphy.get_celldb_file(stack.meta['batch'], stack.meta['cellid'], fs=100, stimfmt='envelope')
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat, est_files=[file], fs=100, avg_resp=True, filestate=[1])
    stack.append(nm.est_val.crossval, valfrac=0.1)
    stack.append(nm.aux.onset_edges)


def env100enj(stack):
    """
    Loads a 100 Hz BAPHY .mat file with 'envelope' marker using the
    provided cellid and batch. Then compute and replace stim with envelope onsets.

    only loads index of the mat file where the field 'filestate' is 1

    Specifically for batch296 SSA data, in this case 'filestate' == 1 means Jitter On
    """
    file = ut.baphy.get_celldb_file(stack.meta['batch'], stack.meta['cellid'], fs=100, stimfmt='envelope')
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat, est_files=[file], fs=100, avg_resp=True, filestate=[0])
    stack.append(nm.est_val.crossval, valfrac=0.1)
    stack.append(nm.aux.onset_edges)

    
def parm50x(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=200,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval_old)
    
def parm50(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=200,stimfmt='parm')
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval, valfrac=0.2)

def parm50pt(stack):
    """
    Loads a 100 Hz BAPHY .mat file, extracted pertrial with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16,pertrial=True)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval, valfrac=0.2)
    
def parm50ptp(stack):
    """
    Loads a 100 Hz BAPHY .mat file, extracted pertrial with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16,pertrial=True)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval, valfrac=0.2,keep_filestate=[0])
    
def parm100pt(stack):
    """
    Loads a 100 Hz BAPHY .mat file, extracted pertrial with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Does not average over 
    response rasters, instead treating each trial as a separate stimulus. Applies 
    a 5% estimation/validation split if the est/val datasets are not specified in 
    the file. 
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16,pertrial=True)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.est_val.crossval, valfrac=0.2)
    
def parm50a(stack):
    """
    Loads a 16 channel, 100 Hz BAPHY .mat file with 'parm' marker using the 
    provided cellid and batch, and downsamples it to 50 Hz. Averages the response 
    to each stimulus over its respective raster, and applies a 5% 
    estimation/validation split if the est/val datasets are not specified in 
    the file.
    
    Specifically for batch293 tone-pip data.
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=True)
    stack.append(nm.est_val.crossval,valfrac=0.05)

def fb24ch200(stack):
    """
    Loads a 24 channel, 200 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=200,avg_resp=True)
    stack.append(nm.est_val.standard)
    
def fb24ch100(stack):
    """
    Loads a 24 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=200,stimfmt='ozgf',chancount=24)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True) #Data not preprocessed to 100 Hz, internally converts
    stack.append(nm.est_val.standard)
    
def fb18ch100(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stack.append(nm.est_val.standard)
    
def fb93ch100(stack):
    """
    Loads a 93-channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 

    This is for DIRECT comparison with Sam N-H's cochlear model.
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=93)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stack.append(nm.est_val.standard)
    
def ctx100ch100(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stimdata=ut.io.load_nat_cort(100,stack.data[-1][0]['prestim'],stack.data[-1][0]['duration'],stack.data[-1][0]['poststim'])
    for d in stack.data[-1]:
        d['stim']=stimdata['stim']
    stack.append(nm.est_val.standard)

def coch93ch100(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stimdata=ut.io.load_nat_coch(100,stack.data[-1][0]['prestim'],stack.data[-1][0]['duration'],stack.data[-1][0]['poststim'])
    for d in stack.data[-1]:
        d['stim']=stimdata['stim']
    stack.append(nm.est_val.standard)
    
def fb18ch100x(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=True)
    stack.append(nm.est_val.crossval)
    stack.modules[-1].do_plot=ut.plot.plot_spectrogram
    
def fb18ch100u(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch.
    Does not average over response rasters, instead treating each trial as a separate
    stimulus. Applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=100,avg_resp=False)
    stack.append(nm.est_val.crossval)
      
def fb18ch50(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch,
    then downsamples to 50 Hz.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=True)
    stack.append(nm.est_val.standard)

def fb18ch50u(stack):
    """
    Loads an 18 channel, 100 Hz BAPHY .mat file using the provided cellid and batch,
    then downsamples to 50 Hz.
    Averages the response to each stimulus over its respective raster, and
    applies a 5% estimation/validation split if the est/val datasets are not 
    specified in the file. 
    """
    file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval)

def ecog25(stack):
    stack.append(nm.loaders.load_gen, load_fun='load_ecog')
    stack.append(nm.est_val.crossval,valfrac=0.2)
    stack.modules[-1].do_plot=ut.plot.plot_spectrogram

def loadlocal(stack):
    """
    This keyword is just to load up a local file that is not yet on the BAPHY database.
    Right now just loads files from my computer --njs, June 27 2017
    """
    file='/Users/HAL-9000/Desktop/CompNeuro/batch'+str(stack.meta['batch'])+'/'+str(stack.meta['cellid'])+'_b'+str(stack.meta['batch'])+'_ozgf_c18_fs100.mat'
    #file=ut.baphy.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],fs=100,stimfmt='ozgf',chancount=18)
    print("Initializing load_mat with file {0}".format(file))
    stack.append(nm.loaders.load_mat,est_files=[file],fs=50,avg_resp=False)
    stack.append(nm.est_val.crossval)

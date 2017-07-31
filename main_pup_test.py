#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:02:27 2017

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
"""
filelist=os.listdir('/auto/users/shofer/data/batch294')
files=[]
for i in filelist:
    f=i.split('_')
    files.append(f[0])
"""
#stack.meta['batch']=294
#stack.meta['cellid']='eno022e-b1'
"""modlist=['nopupgain','pupgain','polypupgain02','polypupgain03','polypupgain04','exppupgain','logpupgain',
         'butterworth01','butterworth02','butterworth03','butterworth04','poissonpupgain']"""
#BOL006b-43-1
stack=mn.fit_single_model('BOL006b-43-1', 293, 'parm50_wc03_fir15_nopupgain_dexp_fit02', autoplot=True)
alldata=stack.data
#stack.plot_dataidx=0
#stack.plot_stimidx=10
#stack.quick_plot()
#stack=mn.load_single_model('bbl031f-a1', 291, 'fb18ch50_wc03_fir10_dexp_fit00')
#alldata=stack.data
#din=stack.modules[3].d_in
#print(stack.data.__len__())
#print(stack.modules[-1].d_in[0]['stim'].shape)
#stack=mn.fit_single_model('eno052d-a1', 294, 'perfectpupil50_polypupgain03_fit02', autoplot=True)
#print(stack.data[3][1]['stim'].shape)
#alldata=stack.data
#alldata=stack.data
#stack=mn.fit_single_model('eno052d-a1', 294, 'perfectpupil50_nopupgain_fit01', autoplot=True,crossval=False)
#print(slist.__len__())
#dat1=slist[1].data
#dat2=slist[19].data[-1][1]['repcount'].shape[0]

#valarr=np.split(slist[0].data[-1][1]['stim'],slist[0].data[-1][1]['repcount'].shape[0],0)
#for i in slist[1:]:
    #valarr=np.append(valarr,i.data[-1][1]['stim'],axis=0)
    
#cellid='eno052b-c1'
#batch=293
#modelname="parm50_wc03_fir10_dexp_fit02"

"""
stack=ns.nems_stack()
cellid='eno052b-c1'
batch=293
modelname="parm50_wc03_fir10_dexp_fit02"

stack.meta['batch']=batch
stack.meta['cellid']=cellid
stack.meta['modelname']=modelname

file=bu.get_celldb_file(stack.meta['batch'],stack.meta['cellid'],
                                     fs=100,stimfmt='parm',chancount=16)
print("Initializing load_mat with file {0}".format(file))
stack.append(nm.load_mat,est_files=[file],fs=50,avg_resp=True)
stack.append(nm.crossval,valfrac=0.05)
#stack.append(nm.weight_channels,num_chans=3)
stack.append(nm.fir_filter,num_coefs=10)
nk.fir_mini_fit(stack)
stack.append(nm.nonlinearity,nltype='dexp',fit_fields=['phi'],phi=[1,1,1,1])
#stack.append(nm.nonlinearity,nltype='exp',fit_fields=['phi'],phi=[1,1])
#stack.append(nm.nonlinearity,nltype='poly',fit_fields=['phi'],phi=[0,1,0])
mseidx=nu.find_modules(stack,'mean_square_error')
if not mseidx:
   # add MSE calculator module to stack if not there yet
   stack.append(nm.mean_square_error)
        
   # set error (for minimization) for this stack to be output of last module
   stack.error=stack.modules[-1].error
        
stack.evaluate(2)

stack.fitter=nf.basic_min(stack)
stack.fitter.tol=0.0001
stack.fitter.do_fit()
nk.create_parmlist(stack)
stack.popmodule() #pop MSE 
#print(stack.error.name)

stack.valmode=True
    
    #stack.nests=1

stack.evaluate(1)
 
alldata=stack.data   
fits=stack.parm_fits
print(stack.modules)

stack.append(nm.mean_square_error)
    
corridx=nu.find_modules(stack,'correlation')
if not corridx:
  # add MSE calculator module to stack if not there yet
  stack.append(nm.correlation)
  
print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
                 stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
#if valdata: 
    #stack.plot_dataidx=valdata[0]
#else:
stack.plot_dataidx=3

alldata=stack.data      
    # edit: added autoplot kwarg for option to disable auto plotting
    #       -jacob, 6/20/17
stack.quick_plot()
    
#alldata=copy.deepcopy(stack.data[0])
#allmods=copy.deepcopy(stack.modules)
"""
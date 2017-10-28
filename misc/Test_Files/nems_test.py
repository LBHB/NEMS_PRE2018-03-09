#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import imp
import scipy.io
import pkgutil as pk

import nems.modules as nm
import nems.main as main
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities as ut
import nems.stack as ns

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

imp.reload(nm)
imp.reload(main)
imp.reload(nf)
imp.reload(nk)
imp.reload(ut)
imp.reload(ns)

try:
    import nems.db as nd
    db_exists = True
except Exception as e:
    # If there's an error import nems.db, probably missing database
    # dependencies. So keep going but don't do any database stuff.
    print("Problem importing nems.db, can't update tQueue")
    print(e)
    db_exists = False

#datapath='/Users/svd/python/nems/ref/week5_TORCs/'
#est_files=[datapath + 'tor_data_por073b-b1.mat']

#datapath='/auto/data/code/nems_in_cache/batch271/'
#est_fieles=[datapath + 'chn020f-b1_b271_ozgf_c24_fs200.mat']
#datapath='/Users/svd/python/nems/misc/ref/'
#est_files=[datapath + 'bbl031f-a1_nat_export.mat']
#'/auto/users/shofer/data/batch291/bbl038f-a2_nat_export.mat'
def dexp_fn(phi,X):
    Y=phi[0,0]-phi[0,1]*np.exp(-np.exp(phi[0,2]*(X-phi[0,3])))
    return(Y)

doval=1

if 0:
    """ NAT SOUND """
    cellid="bbl031f-a1"
    #cellid='bbl034e-a1'
    #cellid='bbl070i-a1'
    cellid="bbl031f-a1"
    batch=291  # IC
    
    #cellid="bbl031f-a1"
    #batch=271 #A1
    #modelname="fb18ch100_wcg01_fir15_fit01"
    #modelname="fb24ch100_wcg01_fir15_fit01"
    modelname="fb93ch100_dlog2_wcg02_fir15_fit01"
    #modelname="fb18ch100_wc01_fir15_fit01"
    #modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit01"
    #modelname="fb18ch100x_wc01_stp2pc_fir15_dexp_fit01"
    #cellid="eno052d-a1"
    #batch=294
    #modelname="perfectpupil50_pupgain_fit01"
if 1:
    """ TORC-TONE """
    cellid="TAR010c-06-1"
    batch=301 #A1
    modelname="fb18ch100x_wcg02_fir15_fit01_nested5"

if 0:
    cellid='gus019d-b2'
    batch=289 #A1
    modelname="fb18ch50u_wcg01_fir10_pupgain_fit03_nested5"

# ecog test
if 0:
    channel=38
    cellid="sam-{0:03d}".format(channel)
    batch=300 #ECOG
    #modelname="ecog25_wcg01_fir15_fit03_nested5"
    modelname="ecog25_wcg01_fir15_dexp_fit01_nested5"
    modelname="ecog25_wcg01_fir15_logsig_fit01_nested5"
    
    
""" pupil gain test -- PPS data """
if 0:
    #cellid='gus021d-a2'
    #cellid='gus021d-b1'
    #cellid="BOL006b-11-1"
    cellid="eno053d-c1"
    #cellid="eno048g-b1"
    #cellid="eno054c-b2"
    batch=293
    # "OLD" noah cross val
    #modelname="parm50x_wcg01_fir10_pupgain_fit01_nested5"
    # "IMPROVED" svd cross val
    #modelname="parm50_wcg01_fir10_pupgain_fit01_nested5"
    #modelname="parm50_wcg01_fir10_pupgainctl_fit01_nested5"
    modelname="parm50_wcg01_fir10_pupwgtctl_fit01_nested2"
    #modelname="parm50_wcg01_fir10_pupgain_fit01_nested5"
    #modelname="parm50_wcg01_fir10_pupgainctl_fit01_nested5"

""" pupil gain test -- 2 x VOC data """
if 0:
    cellid="eno052d-a1"
    #cellid="eno023c-c1"
    batch=294
    modelname="perfectpupil50_pupgain_fit01_nested5"
    #modelname="perfectpupil50_pupgainctl_fit01_nested5"
    #modelname="perfectpupil50x_pupgain_fit01_nested5"

""" SSA test """
if 0:
    #cellid='gus018d-d1'
    cellid="gus021c-b2"
    cellid='gus018d-d1'
    batch=296
    modelname="env100e_fir20_fit01_ssa"
    #modelname="env100e_fir20_dexp_fit01"

# following is equivalent of 
#stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)

if 0:
    stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)
else:
    stack=ns.nems_stack(cellid=cellid,batch=batch,modelname=modelname)
    stack.valmode=False
    stack.keyfuns=nk.keyfuns
    
    # extract keywords from modelname, look up relevant functions in nk and save
    # so they don't have to be found again.
    
    # evaluate the stack of keywords    
    if 'nest' in stack.keywords[-1]:
        # special case if last keyword contains "nested". TODO: better imp!
        print('Evaluating stack using nested cross validation. May be slow!')
        k=stack.keywords[-1]
        stack.keyfuns[k](stack)
    else:
        print('Evaluating stack')
        for k in stack.keywords:
            stack.keyfuns[k](stack)

    if doval:
        # validation stuff
        stack.valmode=True
        stack.evaluate(1)
        
        stack.append(nm.metrics.correlation)
        
        #print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
        #             stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
        valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
        if valdata:
            stack.plot_dataidx=valdata[0]
        else:
            stack.plot_dataidx=0

    #nlidx=nems.utilities.utils.find_modules(stack,'nonlin.gain')
    #stack.modules[nlidx[0]].do_plot=nems.utilities.utils.io_scatter_smooth
    stack.quick_plot()
    
    if 0:
        filename = ut.io.get_file_name(cellid, batch, modelname)
        ut.io.save_model(stack, filename)
        preview_file = stack.quick_plot_save(mode="png")
        print("Preview saved to: {0}".format(preview_file))
        if db_exists:
            queueid = None
            r_id = nd.save_results(stack, preview_file, queueid=queueid)
            print("Fit results saved to NarfResults, id={0}".format(r_id))

#stack.modules[1].nests=5
#stack.modules[1].valfrac=0.2
#stack.evaluate_nested()

'''

stack=ns.nems_stack()

stack.meta['batch']=291
#stack.meta['cellid']='chn020f-b1'
#stack.meta['cellid']='bbl031f-a1'
stack.meta['cellid']='bbl061h-a1'
#stack.meta['cellid']='bbl038f-a2_nat_export'

#stack.meta['batch']=267
#stack.meta['cellid']='ama024a-21-1'
stack.meta['batch']=293
stack.meta['cellid']='eno052b-c1'


# add a loader module to stack
#nk.fb18ch100(stack)
nk.parm100(stack)
#nk.loadlocal(stack)

#nk.ev(stack)
stack.append(nm.crossval, valfrac=0.00)

# add fir filter module to stack & fit a little
#nk.dlog(stack)
#stack.append(nm.normalize)
#nk.dlog(stack)
nk.wc02(stack)
nk.fir15(stack)

# add nonlinearity and refit
#nk.dexp(stack)

# following has been moved to nk.fit00
stack.append(nm.mean_square_error,shrink=0.5)
stack.error=stack.modules[-1].error


stack.fitter=nf.fit_iteratively(stack,max_iter=5)
#stack.fitter.sub_fitter=nf.basic_min(stack)
stack.fitter.sub_fitter=nf.coordinate_descent(stack,tol=0.001,maxit=10)
stack.fitter.sub_fitter.step_init=0.05

stack.fitter.do_fit()

stack.valmode=True
stack.evaluate(1)
corridx=nems.utilities.utils.find_modules(stack,'correlation')
if not corridx:
    # add MSE calculator module to stack if not there yet
    stack.append(nm.correlation)    

stack.plot_dataidx=1

# default results plot
stack.quick_plot()

# save
#filename="/auto/data/code/nems_saved_models/batch{0}/{1}.pkl".format(stack.meta['batch'],stack.meta['cellid'])
#nems.utilities.utils.save_model(stack,filename)


## single figure display
#plt.figure(figsize=(8,9))
#for idx,m in enumerate(stack.modules):
#    plt.subplot(len(stack.modules),1,idx+1)
#    m.do_plot()
    
## display the output of each module in a separate figure
#for idx,m in enumerate(stack.modules):
#    plt.figure(num=idx,figsize=(8,3))
#    #ax=plt.plot(5,1,idx+1)
#    m.do_plot(idx=idx)
    
'''

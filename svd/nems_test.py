#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import logging
log = logging.getLogger(__name__)

from time import time
import imp
import scipy.io
import pkgutil as pk

import nems.modules as nm
import nems.main as main
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities as nu
import nems.stack as ns
from nems.keyword.registry import keyword_registry

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal

imp.reload(nm)
imp.reload(main)
imp.reload(nf)
imp.reload(nk)
imp.reload(nu)
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
    """ RDT """
    cellid="oys022b-b1"
    modelname="fb18ch100pt_wcg01_fir15_dexp_fit01"
    batch=269

if 1:
    """ NAT SOUND """
    #cellid="bbl031f-a1"
    #cellid='bbl034e-a1'
    #cellid='bbl070i-a1'
    #batch=291  # IC

    cellid="zee015e-04-1"
    #cellid="chn020f-b1"
    #cellid="eno052b-c1"
    #cellid='chn020f-b1'
    #cellid='chn029d-a1'
    #cellid='TAR010c-21-1'
    batch=271 #A1
    #modelname="fb18ch100_wc01_fir15_fit01"
    
    modelname="fb18ch100_wcg01_fir15_fit01"
    #modelname="fb18ch100_wcg01_fir15_fititer01"
    
    #modelname="fb18ch100_wcg01_fir15_fitannl00"
    #modelname="ctx100ch100_dlog_wc02_fir15_fit01"
    #modelname="fb24ch100_wcg01_fir15_fit01"
    #modelname="fb93ch100_dlog2_wcg02_fir15_fit01"
    #modelname="fb18ch100_wc01_fir15_fit01"
    #modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit01"
    #modelname="fb18ch100x_wc01_stp2pc_fir15_dexp_fit01"
    #cellid="eno052d-a1"
    #batch=294
    #modelname="perfectpupil50_pupgain_fit01"
    
if 0:
    """ TORC-TONE """
    #cellid="bbl074g-a1"
    #batch=303 #PTD IC pupil
    cellid="TAR010c-30-1"
    batch=301 #PTD A1 pupil
    #modelname="parm100pt_wcg02_fir15_behgain_fit01"
    modelname="parm100pt_wcg01_fir15_behgainctl_fit01"
    #modelname="parm100_psth_pupgain_fit01"

if 0:
    """ BVT """
    cellid="gus006b-a2"
    batch=302 #A1
    modelname="parm50pt_wcg01_fir15_stategain_fit01"

if 0:
    cellid='gus019d-b2'
    batch=289 #A1
    modelname="fb18ch50u_wcg01_fir10_pupgain_fit03"

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
    cellid="BOL006b-07-1"
    #cellid="eno048g-b1"
    #cellid="eno054c-b2"
    batch=293

    #modelname="parm50_wcg01_fir10_pupgainctl_fit01"
    #modelname="parm50_wcg01_fir10_pupgainctl_fit01_nested5"
    modelname="parm50_wcg01_fir10_pupwgtctl_fit01"
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

""" SPN test """
if 0:
    #cellid='gus018d-d1'
    cellid="eno024d-b1"
    #cellid="por016d-a1"
    batch=259
    modelname="env100_wcc02_stp1pc_fir15_dexp_fit01"
    #modelname="env100_dlog_fir15_dexp_fit03"

""" Fitter comparisons """
if 0:
    # matched np.org:
    batch = 271
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fit01"
    #cellid='chn029d-a1'; modelname="fb18ch100_wc01_fir15_fit01"
    #cellid='TAR010c-21-1'; modelname="fb18ch100_wc01_fir15_fit01"

    # possible problems?:
    # MSE about 10% higher and r values about 10% lower than np version
    #cellid='chn029d-a1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit01"
    # But this one matched np exactly with same model, diff cell
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit01"
    # Mostly the same for this one as well
    # cellid='eno025c-c1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit01"


    # trying skopt
    # Looking at the intermediate vector outputs, the skopt fitters seem
    # to jump to very large values very quickly for some reason, unlike scipy.
    # (i.e. if phi0 is [0.1, 0.2, 0.1] eval #10 is suddenly [374, -988, 47]
    #  with MSE of 10000000)
    # overall performed equal to scipy at best, and often performed worse.
    # also took much longer to fit.
    # changing n_calls, xi (~maxit) and kappa settings didn't change
    # performance, and had little effect on speed.

    # gp_minimize: mse 0.505, est 0.501, val 0.616
    # forest_minimize: mse 0.505, est 0.501, val 0.617
    # gbrt_minimize: mse 0.505, est 0.501, val 0.616
    # fit02: mse 0.504, est 0.501, val 0.616 (and much faster)
    cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fit01"
    # gp_minimize: MSE 0.615, r_est 0.629, r_val 0.826
    # forest_minimize: mse 0.615, est 0.629, val 0.826
    # gbrt_minimize: mse 0.6204, est 0.626, val 0.816
    # fit02: mse 0.526, est 0.682, val 0.835 (and much faster)
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_skopt02"


    # trying coordinate descent
    # overall: works about the same as basic_min for simple models, but
    #          basic_min works better for the more complicated models that
    #          include nonlinearity.
    # performance about the same as basic_min so far, but usually faster
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fitcoord00"
    # CD no cache: est 0.647, val 0.853, MSE 0.580, t 66s
    # CD yes cache: smaller perf, and time different less noticeable
    #               must be some issue with code, perf shouldn't be different
    # fit02: est 0.682, val 0.836, MSE 0.526, t 23.5s
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fitcoord00"
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit02"


    # trying iterative fit
    # fititer00 (cd): mse: 0.5427, est 0.500, val 0.616
    # fititer01 (basic_min): mse 0.537322028954, est 0.500, val 0.618
    # fit02: mse 0.504, est 0.501, val 0.616
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fititer01"
    # fititer00 (cd): mse: 0.686, est 0.647, val 0.835
    # fititer01 (basic_min): est 0.672, val 0.814, MSE 0.645
    # fit02: est 0.682, val 0.836, MSE 0.526, t 23.5s
    # fit01: same as 02
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fititer00"
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit01"


    # trying fit by type
    # fittype00: est 0.501, val 0.614, mse 0.504
    # fit01/02: est 0.501, val 0.616, mse 0.504
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fittype00"
    # fittype00: est 0.647, val 0.835, mse 0.689
    # fit01/02: est 0.682, val 0.836, MSE 0.526
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fititer00"


    # triyng new BestMatch fitter (related to fit by type)
    # fit01/02: est 0.501, val 0.616, mse 0.504
    # BestMatch: est 0.501, val 0.616, mse 0.504
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fitbest00"
    # BestMatch (limited to 1 iter): est 0.683, val 0.838, mse 0.5241698409...
    #    fitters used:
    #       Fitter used for filters.weight_channels was: coordinate_descent
    #       Fitter used for filters.stp was: anneal_min
    #       Fitter used for filters.fir was: basic_min
    #       Fitter used for nonlin.gain was: anneal_min
    # fit01/02: est 0.682, val 0.835, MSE 0.5257375579...
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fitbest00"
    # Best Match (limited to 1 iter): est 0.711, val 0.812, mse 0.37648381
        #Fitter used for filters.weight_channels was: anneal_min
        #Fitter used for filters.fir was: anneal_min
        #Fitter used for nonlin.gain was: anneal_min
    # fit01: est 0.711, val 0.812, mse 0.376476
    #cellid='chn029d-a1'; modelname='fb18ch100_wcg02_fir15_dexp_fit01'


    # trying new SequentialFit fitter (related to fit iter and CD)
    # fitseq: mse 0.505, est 0.501, val 0.617
    # fit01/02: est 0.501, val 0.616, mse 0.504
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fitseq00"

    # fit02 same performance but 3-5x as fast (SQLP)
    # ah.. but seems that was just b/c the tolerance was less precise
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fit01"
    #cellid='TAR010c-21-1'; modelname="fb18ch100_wc01_fir15_fit01"
    #cellid='chn029d-a1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit02"
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit02"
    #cellid='eno025c-c1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit02"

    # 'Nelder-Mead' (temp changed fit02) super duper slow for no performance gain
    #cellid='chn020f-b1'; modelname="fb18ch100_wc01_fir15_fit01"
    #cellid='TAR010c-21-1'; modelname="fb18ch100_wc01_fir15_fit01"
    #cellid='chn029d-a1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit02"
    #cellid='TAR010c-13-1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit02"
    #cellid='eno025c-c1'; modelname="fb18ch100_wcg01_stp1pc_fir15_dexp_fit02"

    #batch = 259
    # this one looks normal, but...
    #cellid='por077a-c1'; modelname='env100e_dlog_stp1pc_fir15_dexp_fit02'
    # ** issue with this one? plots are screwey. same with both fitters
    #cellid='chn079d-b1'; modelname='env100e_dlog_stp1pc_fir15_dexp_fit01'
    # ** same with this one
    #cellid='sti030b-d1'; modelname='env100e_dlog_stp1pc_fir15_dexp_fit01'

    #batch = 302
    # can't find data for this
    #cellid='gus027b-a1'; modelname='parm50pt_wcg02_fir15_dexp_fit01'


# following is equivalent of
#stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)


if 0:
    stack=main.fit_single_model(cellid, batch, modelname,autoplot=False)
else:
    stack=ns.nems_stack(cellid=cellid,batch=batch,modelname=modelname)
    stack.valmode=False
    #stack.keyfuns=nk.keyfuns

    # extract keywords from modelname, look up relevant functions in nk and save
    # so they don't have to be found again.

    # evaluate the stack of keywords
    if 'nested' in stack.keywords[-1]:
        # special case for nested keywords. Stick with this design?
        print('Using nested cross-validation, fitting will take longer!')
        k = stack.keywords[-1]
        keyword_registry[k](stack)
    else:
        print('Using standard est/val conditions')
        for k in stack.keywords:
            log.info(k)
            start = time()
            keyword_registry[k](stack)
            end = time()
            elapsed = end-start
            if k == stack.keywords[-1]:
                print("Time to add and run fit: %s seconds."%elapsed)

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
        filename = nu.io.get_file_name(cellid, batch, modelname)
        nu.io.save_model(stack, filename)
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

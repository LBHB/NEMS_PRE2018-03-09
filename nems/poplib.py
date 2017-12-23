#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 23:16:23 2017

@author: svd
"""

import logging
log = logging.getLogger(__name__)

import scipy.io

import nems.modules as nm
import nems.main as main
import nems.fitters as nf
import nems.keyword as nk
import nems.utilities as nu
import nems.stack as ns
import nems.db as ndb
from nems.keyword.registry import keyword_registry

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import copy

def factor_strf_fit(site='TAR010c16', factorN=0, batch=271, modelname="fchan100_wc02_fir15_fit01"):
    #site='zee015h05'
    doval=1
    cellid="{0}-F{1}".format(site,factorN)
    
    # load the stimulus
    stack=ns.nems_stack(cellid=cellid,batch=batch,modelname=modelname)
    stack.meta['resp_channels']=[factorN]
    stack.meta['site']=site
    stack.keyfuns=0

    stack.valmode=False
    
    # evaluate the stack of keywords
    if 'nested' in stack.keywords[-1]:
        # special case for nested keywords. Stick with this design?
        print('Using nested cross-validation, fitting will take longer!')
        k = stack.keywords[-1]
        keyword_registry[k](stack)
    else:
        print('Using standard est/val conditions')
        for k in stack.keywords:
            print(k)
            keyword_registry[k](stack)

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
    
    stack.quick_plot()
    
    savefile = nu.io.get_file_name(cellid, batch, modelname)
    nu.io.save_model(stack, savefile)

    return stack

def factor_strf_load(site='TAR010c16', factorN=0, batch=271, modelname="fchan100_wc02_fir15_fit01"):
    #site='zee015h05'
    cellid="{0}-F{1}".format(site,factorN)
    
    # load the stimulus
    stack=nu.io.load_single_model(cellid, batch, modelname, evaluate=True)

    return stack


def pop_factor_strf_init(site='TAR010c16',factorCount=4,batch=271,fmodelname="fchan100_wc02_fir15_fit01"):

    # find all cells in site that meet iso criterion
    d=ndb.get_batch_cells(batch=batch,cellid=site[:-2])
    d=d.loc[d['min_isolation'] >=75]
    d=d.loc[d['cellid'] != 'TAR010c-21-2']
    d.reset_index(inplace=True)
    
    cellcount=len(d['cellid'])
    
    # modelname should be compatible with fmodelname
    modelname=fmodelname.replace("fchan100","ssfb18ch100")
    modelname=modelname.replace("_fit01","")
    stack=ns.nems_stack(cellid=site,batch=batch,modelname=modelname)
    stack.meta['site']=site
    stack.meta['d']=d
    stack.meta['factorCount']=d
    stack.keyfuns=0
    stack.valmode=False
    
    # evaluate the stack of keywords
    if 'nested' in stack.keywords[-1]:
        # special case for nested keywords. Stick with this design?
        print('Using nested cross-validation, fitting will take longer!')
        k = stack.keywords[-1]
        keyword_registry[k](stack)
    else:
        print('Using standard est/val conditions')
        for k in stack.keywords:
            print(k)
            keyword_registry[k](stack)
    
    wc0=nu.utils.find_modules(stack,'filters.weight_channels')[0]
    fir0=nu.utils.find_modules(stack,'filters.fir')[0]
    stack.modules[fir0].baseline[0,0]=0
    stack.modules[fir0].fit_fields=['coefs']
    
    # load strfs fit to factors
    factorN=0
    cellid="{0}-F{1}".format(site,factorN)
    savefile = nu.io.get_file_name(cellid, batch, fmodelname)
    tstack = nu.io.load_model(savefile)
    
    stack.modules[wc0].phi=tstack.modules[wc0].phi
    stack.modules[wc0].coefs=tstack.modules[wc0].coefs
    stack.modules[wc0].baseline=tstack.modules[wc0].baseline
    stack.modules[wc0].num_chans=stack.modules[wc0].phi.shape[0]
    
    stack.modules[fir0].coefs=tstack.modules[fir0].coefs
    stack.modules[fir0].num_dims=stack.modules[fir0].coefs.shape[0]
    stack.modules[fir0].bank_count=1
    
    for factorN in range(1,factorCount):
        cellid="{0}-F{1}".format(site,factorN)
        savefile = nu.io.get_file_name(cellid, batch, fmodelname)
        tstack = nu.io.load_model(savefile)
        
        stack.modules[wc0].phi=np.concatenate((stack.modules[wc0].phi,
                     tstack.modules[wc0].phi),axis=0)
        stack.modules[wc0].coefs=np.concatenate((stack.modules[wc0].coefs,
                                                 tstack.modules[wc0].coefs),axis=0)
        stack.modules[wc0].baseline=np.concatenate((stack.modules[wc0].baseline,
                     tstack.modules[wc0].baseline),axis=0)
        stack.modules[wc0].num_chans=stack.modules[wc0].phi.shape[0]
        
        stack.modules[fir0].coefs=np.concatenate((stack.modules[fir0].coefs,
                     tstack.modules[fir0].coefs),axis=0)
        stack.modules[fir0].num_dims=stack.modules[fir0].coefs.shape[0]
        stack.modules[fir0].bank_count+=1
    
    stack.evaluate(1)
    stack.append(nm.filters.WeightChannels,num_chans=cellcount)
    stack.modules[-1].fit_fields=['coefs','baseline']
    stack.modules[-1].coefs[:,:]=0
    
    stack.append(nm.metrics.mean_square_error)
    stack.error=stack.modules[-1].error
    
    stack.evaluate(2)
    
    return stack

def pop_factor_strf_fit(stack):
    
    site=stack.meta['site']
    factorCount=stack.meta['factorCount']
    batch=stack.meta['batch']
    modelname=stack.meta['modelname']
    
    stack.fitter=nf.fitters.basic_min(stack)
    
    # first just fit weights at the end
    wcidx=nu.utils.find_modules(stack,'filters.weight_channels')
    firidx=nu.utils.find_modules(stack,'filters.fir')
    stack.fitter.fit_modules=[wcidx[1]]
    
    stack.fitter.tolerance=0.0001
    stack.fitter.do_fit()
    
    stack.fitter=nf.fitters.fit_iteratively(stack)
    stack.fitter.module_sets=[[wcidx[0],firidx[0]],[wcidx[1]]]
    stack.fitter.max_iter=4
    stack.fitter.do_fit()
    
    #stack.fitter.tolerance=0.000001
    #stack.fitter.do_fit()
    #
    #stack.fitter.fit_modules=save_fit_modules[0:-1]
    #stack.fitter.tolerance=0.00001
    #stack.fitter.do_fit()
    #
    #stack.fitter.fit_modules=[save_fit_modules[-1]]
    #stack.fitter.tolerance=0.00001
    #stack.fitter.do_fit()
    
    # clean up fit for display
    
    stack.append(nm.metrics.correlation)
    stack.valmode=True
    stack.evaluate(1)
    
    stack.plot_dataidx=1
    stack.plot_stimidx=0
    stack.quick_plot()
    
    savefile="/auto/data/code/nems_saved_models/batch{0}/site_{1}_F{2}_{3}.pkl".format(batch,site,factorCount,modelname)
    nu.io.save_model(stack,savefile)
    
    return stack


def pop_factor_strf_eval(stack, base_modelname="fb18ch100_wcg02_fir15_fit01"):
    
    d=stack.meta['d']
    batch=stack.meta['batch']
    cellcount=len(d)
    
    # pull out pop model preds
    m=stack.modules[-1]
    resp = m.d_in[stack.plot_dataidx]['resp'][:,stack.plot_stimidx,:]
    pred = m.d_in[stack.plot_dataidx]['pred'][:,stack.plot_stimidx,:]
    wcidx=nu.utils.find_modules(stack,'filters.weight_channels')
    firidx=nu.utils.find_modules(stack,'filters.fir')
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.imshow(stack.modules[wcidx[-1]].coefs,aspect='auto')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(pred,aspect='auto')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(resp,aspect='auto')
    plt.colorbar()
    
    iso_rval=np.zeros([cellcount,1])
    for cellnum in range(0,cellcount):
        cellid=d['cellid'][cellnum]
        tstack=nu.io.load_single_model(cellid=cellid,modelname="fb18ch100_wcg02_fir15_fit01",batch=batch)
        m2=tstack.modules[-1]
        if cellnum==0:
            iso_resp2 = m2.d_in[tstack.plot_dataidx]['resp']
            iso_pred2 = m2.d_in[tstack.plot_dataidx]['pred']
        else:
            iso_resp2=np.concatenate((iso_resp2,m2.d_in[tstack.plot_dataidx]['resp']),axis=0)
            iso_pred2=np.concatenate((iso_pred2,m2.d_in[tstack.plot_dataidx]['pred']),axis=0)
        iso_rval[cellnum]=tstack.meta['r_val']
    
    cellnum=0
    cellid=d['cellid'][cellnum]
    stack.plot_stimidx=0
    resp = m.d_in[stack.plot_dataidx]['resp'][cellnum,stack.plot_stimidx,:]
    pred = m.d_in[stack.plot_dataidx]['pred'][cellnum,stack.plot_stimidx,:]
    
    resp2 = iso_resp2[cellnum,stack.plot_stimidx,:]
    pred2 = iso_pred2[cellnum,stack.plot_stimidx,:]
    
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(pred)
    plt.plot(resp)
    plt.title("{}: rval: {:3.2f}".format(cellid,stack.modules[-1].r_val_perunit[cellnum][0]))
    
    plt.subplot(3,1,2)
    plt.plot(pred2)
    plt.plot(resp2)
    plt.title("{}: rval: {:3.2f}".format(cellid,iso_rval[cellnum][0]))
    
    plt.subplot(3,2,5)
    hiso,=plt.plot(iso_rval,label='isolation')
    hss,=plt.plot(stack.modules[-1].r_val_perunit,label='subspace')
    plt.legend(handles=[hiso,hss])

    plt.subplot(3,2,6)
    plt.plot(np.array([-0.05,0.8]),np.array([-0.05,0.8]),)
    plt.axis('equal')
    plt.plot(iso_rval,stack.modules[-1].r_val_perunit,'k.')
    
    plt.tight_layout()
    return stack

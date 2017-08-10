#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:16:37 2017

@author: svd
"""
import nems.modules as nm
import nems.stack as ns
import nems.utilities as ut
import nems.keywords as nk
import operator as op
import numpy as np

"""
fit_single_model - create, fit and save a model specified by cellid, batch and modelname

example fit on nice IC cell:
    import lib.main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    nems.fit_single_model(cellid,batch,modelname)

"""
def fit_single_model(cellid, batch, modelname, autoplot=True,**xvals): #Remove xvals later, need to rework web app
    """
    Fits a single NEMS model. With the exception of the autoplot feature,
    all the details of modelfitting are taken care of by the model keywords.
    
    fit_single_model functions by iterating through each of the keywords in the
    modelname, and perfroming the actions specified by each keyword, usually 
    appending a nems module. Nested crossval is implemented as a special keyword,
    which is placed last in a modelname/
    
    fit_single_model returns the evaluated stack, which contains both the estimation
    and validation datasets. In the caste of nested crossvalidation, the validation
    dataset contains all the data, while the estimation dataset is just the estimation 
    data that was fitted last (i.e. on the last nest)
    """
    stack=ns.nems_stack()
    
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname
    
    # extract keywords from modelname    
    stack.keywords=modelname.split("_")
    if 'nested' in stack.keywords[-1]:
        print('Using nested cross-validation, fitting will take longer!')
        f=getattr(nk,stack.keywords[-1])
        f(stack)
    else:
        print('Using standard est/val conditions')
        stack.valmode=False
        for k in stack.keywords:
            f = getattr(nk, k)
            f(stack)
            
    # measure performance on both estimation and validation data
    stack.valmode=True
    stack.evaluate(1)
    
    stack.append(nm.metrics.correlation)
                    
    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
                 stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
    valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx=valdata[0]
    else:
        stack.plot_dataidx=0

        
    # edit: added autoplot kwarg for option to disable auto plotting
    #       -jacob, 6/20/17
    if autoplot:
        stack.quick_plot()
    
    # save
    filename = ut.utils.get_file_name(cellid, batch, modelname)
    ut.utils.save_model(stack, filename)

    return(stack)

"""
load_single_model - load and evaluate a model, specified by cellid, batch and modelname

example:
    import lib.nems_main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    stack=nems.load_single_model(cellid,batch,modelname)
    stack.quick_plot()
    
"""
def load_single_model(cellid, batch, modelname):
    
    filename = ut.utils.get_file_name(cellid, batch, modelname)
    stack = ut.utils.load_model(filename)
    
    try:
        stack.valmode = True
        stack.evaluate()
    except Exception as e:
        print("Error evaluating stack")
        print(e)
        # TODO: What to do here? Is there a special case to handle, or
        #       did something just go wrong?
    #stack.quick_plot()
    return stack

def load_from_dict(batch,cellid,modelname):
    filepath = ut.utils.get_file_name(cellid, batch, modelname)
    sdict=ut.utils.load_model_dict(filepath)
    #Maybe move some of this to the load_model_dict function?
    stack=ns.nems_stack()
    #stack.valmode=True
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname
    stack.nests=sdict['nests']
    parm_list=[]
    for i in sdict['parm_fits']:
        parm_list.append(np.array(i))
    stack.parm_fits=parm_list
    stack.cv_counter=sdict['cv_counter']
    stack.fitted_modules=sdict['fitted_modules']
    
    for i in range(0,len(sdict['modlist'])):
        stack.append(op.attrgetter(sdict['modlist'][i])(nm),**sdict['mod_dicts'][i])
        #stack.evaluate()
        
    stack.valmode=True
    stack.evaluate()
    stack.quick_plot()
    return stack











    

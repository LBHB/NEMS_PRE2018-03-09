#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:16:37 2017

@author: svd
"""
import nems.modules as nm
import nems.stack as ns
import nems.utilities
import operator as op
import numpy as np
import pkgutil as pk

from nems.keyword.registry import keyword_registry

"""
fit_single_model - create, fit and save a model specified by cellid, batch and modelname

example usage for one nice IC cell:
    import lib.main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_wcg02_fir15_dexp_fit01'
    nems.fit_single_model(cellid,batch,modelname)

"""


# Remove xvals later, need to rework web app
def fit_single_model(cellid, batch, modelname,
                     autoplot=True, saveInDB=True, **xvals):
    """
    Fits a single NEMS model. With the exception of the autoplot feature,
    all the details of modelfitting are taken care of by the model keywords.

    fit_single_model functions by iterating through each of the keywords in the
    modelname, and perfroming the actions specified by each keyword, usually
    appending a nems module. Nested crossval is implemented as a special keyword,
    which is placed last in a modelname.

    fit_single_model returns the evaluated stack, which contains both the estimation
    and validation datasets. In the caste of nested crossvalidation, the validation
    dataset contains all the data, while the estimation dataset is just the estimation
    data that was fitted last (i.e. on the last nest)
    """
    stack = ns.nems_stack()

    stack.meta['batch'] = batch
    stack.meta['cellid'] = cellid
    stack.meta['modelname'] = modelname
    stack.valmode = False
    stack.keywords = modelname.split("_")

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

    # measure performance on both estimation and validation data
    stack.valmode = True
    stack.evaluate(1)

    stack.append(nm.metrics.correlation)

    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
                                                                  stack.meta['mse_val'], stack.meta['r_est'], stack.meta['r_val']))
    valdata = [i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx = valdata[0]
    else:
        stack.plot_dataidx = 0
    #phi = stack.fitter.fit_to_phi()
    #stack.meta['n_parms'] = len(phi)

    # edit: added autoplot kwarg for option to disable auto plotting
    #       -jacob, 6/20/17
    if autoplot:
        stack.quick_plot()

    # save in data base
    if saveInDB == True:
        filename = nems.utilities.io.get_file_name(cellid, batch, modelname)
        nems.utilities.io.save_model(stack, filename)

    return(stack)

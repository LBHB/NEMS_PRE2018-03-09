#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import scipy.signal as sps
import scipy
import numpy as np

import nems.modules
import nems.fitters


#
# random utilties
#
def find_modules(stack, mod_name):
    matchidx = [i for i, m in enumerate(stack.modules) if m.name == mod_name]
    if not matchidx:
        raise ValueError('Module not present in this stack')
    return matchidx


def shrinkage(mH, eH, sigrat=1, thresh=0):
    smd = np.abs(mH) / (eH + np.finfo(float).eps * (eH == 0)) / sigrat

    if thresh:
        hf = mH * (smd > 1)
    else:
        smd = 1 - np.power(smd, -2)
        smd = smd * (smd > 0)
        # smd[np.isnan(smd)]=0
        hf = mH * smd

    return hf


def concatenate_helper(stack, start=1, **kwargs):
    """
    Helper function to concatenate the nest list in the validation data. Simply
    takes the lists in stack.data if ['est'] is False and concatenates all the
    subarrays.
    """
    try:
        end = kwargs['end']
    except BaseException:
        end = len(stack.data)
    for k in range(start, end):
        # print('start loop 1')
        # print(len(stack.data[k]))
        for n in range(0, len(stack.data[k])):
            # print('start loop 2')
            if stack.data[k][n]['est'] is False:
                # print('concatenating')
                if stack.data[k][n]['stim'][0].ndim == 3:
                    stack.data[k][n]['stim'] = np.concatenate(
                        stack.data[k][n]['stim'], axis=1)
                else:
                    stack.data[k][n]['stim'] = np.concatenate(
                        stack.data[k][n]['stim'], axis=0)
                stack.data[k][n]['resp'] = np.concatenate(
                    stack.data[k][n]['resp'], axis=0)
                try:
                    stack.data[k][n]['pupil'] = np.concatenate(
                        stack.data[k][n]['pupil'], axis=0)
                except BaseException:
                    stack.data[k][n]['pupil'] = None
                try:
                    stack.data[k][n]['replist'] = np.concatenate(
                        stack.data[k][n]['replist'], axis=0)
                except BaseException:
                    stack.data[k][n]['replist'] = []
                try:
                    stack.data[k][n]['repcount'] = np.concatenate(
                        stack.data[k][n]['repcount'], axis=0)
                except BaseException:
                    pass
                if 'stim2' in stack.data[k][n]:
                    if stack.data[k][n]['stim2'][0].ndim == 3:
                        stack.data[k][n]['stim2'] = np.concatenate(
                            stack.data[k][n]['stim2'], axis=1)
                    else:
                        stack.data[k][n]['stim2'] = np.concatenate(
                            stack.data[k][n]['stim2'], axis=0)
            else:
                pass


def thresh_resamp(data, resamp_factor, thresh=0, ax=0):
    """
    Helper function to apply an FIR downsample to data. If thresh is specified,
    the function will send all values in data below thresh to 0; this is often
    useful to reduce the ringing caused by FIR downsampling.
    """
    resamp = sps.decimate(data, resamp_factor, ftype='fir',
                          axis=ax, zero_phase=True)
    mask = np.isfinite(resamp)
    s_indices = resamp[mask] < thresh
    mask[mask] = s_indices
    resamp[mask] = 0
    return resamp


def bin_resamp(data, resamp_factor, ax=0):
    """
    Integer downsampling-- just average values occuring in each group of
    resp_factor bins along axis ax. Gets rid of edge effects and ringing. Plus
    it makes more sense for rebinning single-trial spike rates.
    """
    s = np.array(data.shape)
    snew = np.concatenate(
        (s[0:ax], [resamp_factor], [s[ax] / resamp_factor], s[(ax + 1):]))
    # print(s)
    # print(snew)
    d = np.reshape(data, snew.astype(int), order='F')
    d = np.mean(d, ax)
    return d


def stretch_trials(data):
    """
    Helper function to "stretch" trials to be treated individually as stimuli.
    This function is used when it is not desirable to average over the trials
    of the stimuli in a dataset, such as when the effects of state variables such
    as pupil diameter are being explored.

    'data' should be the imported data dictionary, and must contain 'resp',
    'stim', 'pupil', and 'repcount'. Note that 'stim' should be formatted as
    (channels,stimuli,time), while 'resp' and 'pupil' should be formatted as
    (time,trials,stimuli). These are the configurations used in the default
    loading module nems.modules.load_mat
    """
    # r=data['repcount']
    s = data['resp'].shape  # time X rep X stim

    # stack each rep on top of each other
    resp = np.transpose(data['resp'], (0, 2, 1))  # time X stim X rep
    resp = np.transpose(np.reshape(
        resp, (s[0], s[1] * s[2]), order='F'), (1, 0))

    # data['resp']=np.transpose(np.reshape(data['resp'],(s[0],s[1]*s[2]),order='C'),(1,0)) #Interleave
    # mask=np.logical_not(npma.getmask(npma.masked_invalid(resp)))
    # R=resp[mask]
    # resp=np.reshape(R,(-1,s[0]),order='C')
    try:
        # stack each rep on top of each other -- identical to resp
        pupil = np.transpose(data['pupil'], (0, 2, 1))
        pupil = np.transpose(np.reshape(
            pupil, (s[0], s[1] * s[2]), order='F'), (1, 0))
        # P=pupil[mask]
        # pupil=np.reshape(P,(-1,s[0]),order='C')
        # data['pupil']=np.transpose(np.reshape(data['pupil'],(s[0],s[1]*s[2]),order='C'),(1,0))
        # #Interleave
    except ValueError:
        pupil = None

    # copy stimulus as many times as there are repeats -- same stacking as
    # resp??
    stim = data['stim']
    for i in range(1, s[1]):
        stim = np.concatenate((stim, data['stim']), axis=1)

    # construct list of which stimulus idx was played on each trial
    # should be able to do this much more simply!
    lis = np.mat(np.arange(s[2])).transpose()
    replist = np.repeat(lis, s[1], axis=1)
    replist = np.reshape(replist.transpose(), (-1, 1))

    # find non-nan response trials
    keepidx = np.isfinite(resp[:, 0])
    resp = resp[keepidx, :]
    stim = stim[:, keepidx, :]
    replist = replist[keepidx]
    if not pupil is None:
        pupil = pupil[keepidx, :]

    #    Y=data['stim'][:,0,:]
    #    stim=np.repeat(Y[:,np.newaxis,:],r[0],axis=1)
    #    for i in range(1,s[2]):
    #        Y=data['stim'][:,i,:]
    #        Y=np.repeat(Y[:,np.newaxis,:],r[i],axis=1)
    #        stim=np.append(stim,Y,axis=1)
    #    lis=[]
    #    for i in range(0,r.shape[0]):
    #        lis.extend([i]*data['repcount'][i])
    #    replist=np.array(lis)
    return stim, resp, pupil, replist


def mini_fit(stack, mods=['filters.weight_channels',
                          'filters.fir', 'filters.stp']):
    """
    Helper function that module coefficients in mod list prior to fitting
    all the model coefficients. This is often helpful, as it gets the model in the
    right ballpark before fitting other model parameters, especially when nonlinearities
    are included in the model.

    This function is not appended directly to the stack, but instead is included
    in keywords
    """
    stack.append(nems.modules.metrics.mean_square_error, shrink=0.05)
    stack.error = stack.modules[-1].error
    fitidx = []
    for i in mods:
        try:
            fitidx = fitidx + find_modules(stack, i)
        except BaseException:
            fitidx = fitidx + []
    fitidx.sort()
    stack.fitter = nems.fitters.fitters.basic_min(
        stack, fit_modules=fitidx, tolerance=0.00001)

    stack.fitter.do_fit()
    stack.popmodule()


def create_parmlist(stack):
    """
    Helper function that assigns all fitted parameters for a model to a single (n,)
    phi vector and accociates it to the stack.parm_fits object
    """
    stack.fitted_modules = []
    phi = []
    for idx, m in enumerate(stack.modules):
        this_phi = m.parms2phi()
        if this_phi.size:
            stack.fitted_modules.append(idx)
            phi.append(this_phi)
    phi = np.concatenate(phi)
    stack.parm_fits.append(phi)


def nest_helper(stack, nests=20):
    """
    Helper function for implementing nested cross-validation. Essentially sets up
    a loop with the estimation part of fit_single_model inside.
    """
    stack.meta['cv_counter'] = 0
    stack.meta['nests'] = nests
    # stack.cond=False
    # stack.nests=nests

    while stack.meta['cv_counter'] < nests:
        print('Nest #' + str(stack.meta['cv_counter']))
        stack.clear()

        stack.valmode = False

        for i in range(0, len(stack.keywords) - 1):
            stack.keyfuns[stack.keywords[i]](stack)
            # if stack.modules[-1].name=="est_val.crossval2":
            #    stack.modules[-1].cv_counter=stack.meta['cv_counter']
            #    stack.modules[-1].evaluate()
            # if stack.modules[-1].name=="est_val.crossval":
            #    stack.modules[-1].cv_counter=stack.meta['cv_counter']
            #    stack.nests=stack.meta['nests']
            #    stack.modules[-1].evaluate()
        stack.meta['cv_counter'] += 1

    stack.meta['cv_counter'] = 0
    # stack.cv_counter=0  # reset to avoid problem with val stage


def crossval_set(n_trials, cv_count=10, cv_idx=None,
                 interleave_valtrials=True):
    """ create a list trial indices to save for cross-validation in a nested
    cross-val procedure. standardized so it can be used across different
    analyses. user provides:
        n_trials - total number of trials
        cv_count - number of cross-val sets
        cv_idx - the set 0..(cv_count-1) to return indices. if None, return
            list with indices for all cv_idx values
        interleave_valtrials: validx includes every cv_count-th trial if true
            otherwise, just block by trials (maybe prone to bias from slow
            changes in state)
    returns (estidx,validx) tuple
    """

    spl = n_trials / cv_count

    if n_trials < cv_count:
        raise IndexError(
            'Fewer stimuli than cv_count; cv_count<=n_trials required')

    # figure out grouping for each CV set
    if interleave_valtrials:
        smax = np.int(np.ceil(spl))
        a = np.arange(np.ceil(spl) * cv_count).astype(int)
        a = np.reshape(a, [smax, cv_count])
        a = a.transpose()
        a = np.reshape(a, [smax * cv_count])
        a = a[a < n_trials]
    else:
        a = np.arange(n_trials).astype(int)

    estidx_sets = []
    validx_sets = []
    for cc in range(0, cv_count):
        c1 = np.int(np.floor((cc) * spl))
        c2 = np.int(np.floor((cc + 1) * spl))
        validx_sets.append(a[c1:c2])
        estidx_sets.append(np.setdiff1d(a, validx_sets[-1]))

    if cv_idx is None:
        return (estidx_sets, validx_sets)
    else:
        return (estidx_sets[cv_idx], validx_sets[cv_idx])

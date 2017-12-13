'''

Tools for fitting neural receptive field (NRF) model

created CRH 09_20_2017

'''

import numpy as np
import numpy.linalg as la
import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
sys.path.append('/auto/users/hellerc/nems/nems/utilities')
from utils import crossval_set


def get_weight_mat(r, lag=1, fs=1):
    '''
    Args
    ---------------------------------
    r (numpy array, required): 4-D array of cell responses (bins x reps x stim x cells)
                               or 2-D array where dims: (bins*reps*stim x cells)
    lags (int, optional): Time (s) over which to perform reverse correlation.
                          Default is to only do correlation at current time bin
    fs (int): sampling frequency of the r matrix
    Output
    ---------------------------------
    h (numpy array): 3-D array containing the relative weight between all pairs of
                     neurons (cells X lags X weights) - 2-D if lags not specified
    '''
    if lag != 1 and fs == 1:
        sys.exit('Must specify sampling frequency')

    lags = int(lag * fs)
    if len(r.shape) > 2:
        bincount = r.shape[0]
        repcount = r.shape[1]
        stimcount = r.shape[2]
        cellcount = r.shape[3]
        r = r.reshape(bincount * stimcount * repcount, cellcount)
    else:
        cellcount = r.shape[1]

    h = np.empty((cellcount, lags, cellcount - 1))
    for i in range(0, cellcount):
        neuron = r[:, i]
        r_temp = np.delete(r, i, 1)
        Css = np.matmul(r_temp.T, r_temp) / len(r_temp)
        for j in range(0, lags):
            Csr = np.matmul(r_temp.T, np.roll(neuron, j)) / len(r_temp)
            h_temp = np.matmul(la.inv(Css).T, Csr)
            h[i, j, :] = np.concatenate((h_temp[0:i], h_temp[i:len(h_temp)]))

    return np.squeeze(h)


def get_PSTH(r):
    '''
    Args
    -----------------------------------------
    r (numpy array): 4-D array of cell responses (bins x reps x stim x cells),

    Output
    -----------------------------------------
    psth (numpy array): bincount x stims x cells psth array
    '''
    bincount = r.shape[0]
    repcount = r.shape[1]
    stimcount = r.shape[2]
    cellcount = r.shape[3]

    psth = np.empty((bincount, stimcount, cellcount))
    for stim in range(0, stimcount):
        for cell in range(0, cellcount):
            psth[:, stim, cell] = np.squeeze(np.mean(r[:, :, stim, cell], 1))
    return psth


def NRF_fit(r, r0_strf=None, cv_count=10, lag=1, fs=1,
            model=None, spontonly=None, shuffle=True):
    '''
    Function fits full model (rN + r0), rN only, or r0 only and returns the
    array of predicted responses for the model of choice.

    Args
    -------------------------------------------
    r (numpy array, required): 4-D (or 3-D) array of cell responses (bins x reps x stim x cells)
    r0_strf (numpy array, required if model = NRF_STRF): 3-D array of strf precited responses to PPS stims
    trialset (float): Fraction of data to be used as the trial set. Default is 0.8
    lag (int, optional): Time (s) over which to perform reverse correlation.
                          Default is to only do correlation at current time bin
    fs (int): sampling freq uency of the r matrix
    model (string): String specifying which model you'd like to use for the fit
                            NRF_only
                            PSTH_only
                            NRF_PSTH
    Output
    --------------------------------------------
    r_fit: 4-D arrays of predicted neural responses during test set
    '''
    if spontonly is None:
        sys.exit('must specify if spontonly')
    if model == 'NRF_STRF' and r0_strf is None:
        sys.exit('must provide null model (strf)')
    bincount = r.shape[0]
    repcount = r.shape[1]
    if len(r.shape) == 4:
        stimcount = r.shape[2]
        cellcount = r.shape[3]
    elif len(r.shape) == 3:
        cellcount = r.shape[2]

    psth = get_PSTH(r)
    rdiff = r - np.tile(psth[:, np.newaxis, :, :], [1, repcount, 1, 1])
    train_idx, test_idx = crossval_set(
        repcount, cv_count=cv_count, interleave_valtrials=shuffle)
    prediction = np.empty(r.shape)
    for i in range(0, len(train_idx)):
        # ======================== pulling train/test sets ====================
        r_train = r[:, train_idx[i], :, :]
        rdiff_train = rdiff[:, train_idx[i], :, :]
        r_test = r[:, test_idx[i], :, :]
        rdiff_test = rdiff[:, test_idx[i], :, :]

        pred = np.empty(
            (r_test.shape[0] * r_test.shape[1] * r_test.shape[2], cellcount))
# ============================== fitting models ==============================
        if model == 'NRF_only':
            h = get_weight_mat(r_train)
            for neuron in range(0, cellcount):
                stim = r_test.reshape(
                    bincount * len(test_idx[i]) * stimcount, cellcount)
                stim = np.delete(stim, neuron, 1)
                h_neuron = h[neuron, :]
                pred[:, neuron] = np.matmul(np.squeeze(h_neuron), stim.T)

        elif model == 'PSTH_only':
            pred = get_PSTH(r_train)
            pred = np.tile(pred[:, np.newaxis, :, :], [
                           1, len(test_idx[i]), 1, 1])
            pred = pred.reshape(
                bincount * len(test_idx[i]) * stimcount, cellcount)

        elif model == 'NRF_PSTH':
            if spontonly:
                h = get_weight_mat(r_train)
            else:
                h = get_weight_mat(rdiff_train)
            psth_stim = get_PSTH(r_train)
            psth_stim = np.tile(psth_stim[:, np.newaxis, :, :], [
                                1, len(test_idx[i]), 1, 1])
            psth_stim = psth_stim.reshape(
                bincount * len(test_idx[i]) * stimcount, cellcount)  # making psth prediciton
            for neuron in range(0, cellcount):
                if spontonly:
                    stim = r_test.reshape(
                        bincount * len(test_idx[i]) * stimcount, cellcount)
                else:
                    stim = rdiff_test.reshape(
                        bincount * len(test_idx[i]) * stimcount, cellcount)
                stim = np.delete(stim, neuron, 1)
                h_neuron = h[neuron, :]
                pred[:, neuron] = np.matmul(np.squeeze(
                    h_neuron), stim.T) + psth_stim[:, neuron]
        elif model == 'NRF_STRF':

            if len(r0_strf.shape) > 3:
                r0 = r0_strf[:, 0:len(test_idx[i]), :, :]
                r0 = r0.reshape(
                    bincount * len(test_idx[i]) * stimcount, cellcount)
                r_train = r[:, train_idx[i], :, :] - \
                    r0_strf[:, train_idx[i], :, :]
                r_test = r[:, test_idx[i], :, :] - \
                    r0_strf[:, test_idx[i], :, :]
                h = get_weight_mat(r_train)
                for neuron in range(0, cellcount):
                    stim = r_test.reshape(
                        bincount * len(test_idx[i]) * stimcount, cellcount)
                    stim = np.delete(stim, neuron, 1)
                    h_neuron = h[neuron, :]
                    pred[:, neuron] = np.matmul(
                        np.squeeze(h_neuron), stim.T) + r0[:, neuron]
            else:
                r0 = r0_strf.reshape(bincount * repcount, cellcount)
                r_train = r[:, train_idx[i], :] - r0_strf[:, train_idx[i], :]
                r_test = r[:, test_idx[i], :] - r0_strf[:, test_idx[i], :]
                h = get_weight_mat(r_train)
                for neuron in range(0, cellcount):
                    stim = r_test.reshape(
                        bincount * len(test_idx[i]), cellcount)
                    stim = np.delete(stim, neuron, 1)
                    h_neuron = h[neuron, :]
                    pred[:, neuron] = np.matmul(
                        np.squeeze(h_neuron), stim.T) + r0[:, neuron]
        else:
            sys.exit('specify a model selection')
# =========================== Save predictions ===========================
# ========================== Sort predictions ============================
        prediction[:, test_idx[i], :, :] = pred.reshape(
            bincount, len(test_idx[i]), stimcount, cellcount)
    return prediction


def eval_fit(r, rpred):
    '''
    Given the true dataset and predicted data set, this functio will evalute the
    performance of the prediction using pearson correlation coefficient

    Args:
    -------------------------------------------------
    r (numpy array, required): 4-D array of cell responses (bins x reps x stim x cells)
    rpred (numpy array, required): 4-D array of predicted cell responses (bins x reps x stim x cells)
    Output
    --------------------------------------------------
    coeff (dictionary):
        mean (numpy array): correlation coeff across all trials and stims for each cell
        bytrial: (numpy array): reps x stim x cell array of corr coefs for each trial/stim/cell
    '''
    bincount = r.shape[0]
    repcount = r.shape[1]
    if len(r.shape) == 4:
        stimcount = r.shape[2]
        cellcount = r.shape[3]
    elif len(r.shape) == 3:
        stimcount = 1
        cellcount = r.shape[2]
        r = r[:, :, np.newaxis, :]
        rpred = rpred[:, :, np.newaxis, :]
    coeff = dict()
    coeff['mean'] = np.empty(cellcount)
    for i in range(0, cellcount):
        coeff['mean'][i] = (np.corrcoef(r.reshape(bincount * repcount * stimcount, cellcount)[:, i],
                                        rpred.reshape(bincount * repcount * stimcount, cellcount)[:, i])[0][1])

    coeff['bytrial'] = np.empty((repcount, stimcount, cellcount))
    for cell in range(0, cellcount):
        for rep in range(0, repcount):
            for stim in range(0, stimcount):
                coeff['bytrial'][rep, stim, cell] = np.corrcoef(
                    r[:, rep, stim, cell], rpred[:, rep, stim, cell])[0][1]
    return coeff


def sort_bytrial_voc(r, p, data):
    '''
    takes r, and data object (with resp, pupil, stim info, etc.) sorts into trials defined
    by stim type,
    '''
    bincount = r.shape[0]
    trialcount = r.shape[2]
    cellcount = r.shape[-1]
    stimids = data['stimids']

    stim2 = int(max(stimids))
    repcount = int(trialcount / stim2)
    rbystim = np.empty((bincount, stim2, repcount, cellcount))
    pbystim = np.empty((bincount, stim2, repcount))
    for stim in range(0, stim2):
        inds = [t[0] for t in np.argwhere(stimids == (stim + 1))]
        rbystim[:, stim, :, :] = np.squeeze(r[:, 0, inds, :])
        pbystim[:, stim, :] = np.squeeze(p[:, 0, inds])
    rbystim = np.transpose(rbystim, (0, 2, 1, 3))
    pbystim = np.transpose(pbystim, (0, 2, 1))
    return pbystim, rbystim

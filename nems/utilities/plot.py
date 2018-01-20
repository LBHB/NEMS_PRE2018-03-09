#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utilities.plot.py:  plotting functions called by modules in nems stack

Created on Sep 7 2017

@author: svd
"""

import logging
log = logging.getLogger(__name__)

import scipy.signal as sps
import scipy
import numpy as np
import numpy.ma as npma
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools as itt
import nems.utilities.utils

# set default figsize for pyplots (so we don't have to change each function)
FIGSIZE = (12, 4)


#
# PLOTTING FUNCTIONS
#
# TODO: find some way to get the stimuli to resolve correctly for the pupil model stacks,
# since stack.data[1] will have ~2 stimuli, but it is subsequently reshaped to ~240 stimuli
# ---fixed, see except statement in plot_spectrogram --njs 6 July 2017


def plot_spectrogram(m, idx=None, size=FIGSIZE):
    # Moved from pylab to pyplot module in all do_plot functions, changed plots
    # to be individual large figures, added other small details -njs June 16,
    # 2017
    if idx:
        plt.figure(num=idx, figsize=size)
    out1 = m.d_out[m.parent_stack.plot_dataidx]
    reps = out1['repcount']
    ids = m.parent_stack.plot_stimidx
    r = reps.shape[0]
    lis = []
    for i in range(0, r):
        lis.extend([i] * reps[i])
    new_id = lis[ids]
    if out1[m.output_name].ndim == 3:
        try:
            plt.imshow(out1[m.output_name][:, m.parent_stack.plot_stimidx, :],
                       aspect='auto', origin='lower', interpolation='none')
        except BaseException:
            plt.imshow(out1[m.output_name][:, new_id, :],
                       aspect='auto', origin='lower', interpolation='none')
        cbar = plt.colorbar()
        cbar.set_label('amplitude')
        # TODO: colorbar is intensity of spectrogram/response, units not
        # clearly specified yet
        plt.xlabel('Time')
        plt.ylabel('Channel')
    else:
        s = out1[m.output_name][:, new_id]
        # r=out1['resp'][m.parent_stack.plot_stimidx,:]
        pred, = plt.plot(s, label='Average Model')
        # resp, =plt.plot(r,'r',label='Response')
        plt.legend(handles=[pred])
        # TODO: plot time in seconds
        plt.xlabel('Time Step')
        plt.ylabel('Firing rate (a.u.)')

        # plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))


def plot_stim(m, idx=None, size=FIGSIZE):
    if idx:
        plt.figure(num=str(idx), figsize=size)
    out1 = m.d_out[m.parent_stack.plot_dataidx]
    # c=out1['repcount'][m.parent_stack.plot_stimidx]
    # h=out1[m.output_name][m.parent_stack.plot_stimidx].shape
    # scl=int(h[0]/c)
    s2 = np.squeeze(out1[m.output_name]
                    [:, m.parent_stack.plot_stimidx, :]).transpose()
    log.info(s2.shape)
    plt.plot(s2)
    # plt.title(m.name+': stim #'+str(m.parent_stack.plot_stimidx))


def pred_act_scatter(m, idx=None, size=FIGSIZE):
    if idx:
        plt.figure(num=idx, figsize=size)
    out1 = m.d_out[m.parent_stack.plot_dataidx]
    s = out1[m.output_name][0,m.parent_stack.plot_stimidx, :]
    r = out1['resp'][0,m.parent_stack.plot_stimidx, :]
    plt.plot(s, r, 'ko')
    plt.xlabel("Predicted ({0})".format(m.output_name))
    plt.ylabel('Actual')
    # plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    # plt.title("{0} (r_est={1:.3f}, r_val={2:.3f})".format(m.name,m.parent_stack.meta['r_est'],m.parent_stack.meta['r_val']))
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    if ymin == ymax:
        ymax = ymin + 1
    if xmin == xmax:
        xmax = xmin + 1
    # log.info("{0},{1} {2},{3}".format(xmin,xmax,ymin,ymax))
    plt.text(xmin + (xmax - xmin) / 50, ymax - (ymax - ymin) / 20,
             "r_est={0:.3f}\nr_val={1:.3f}".format(m.parent_stack.meta['r_est'][0], m.parent_stack.meta['r_val'][0]),
             verticalalignment='top')


def io_scatter_smooth(m, idx=None, size=FIGSIZE):
    if idx:
        plt.figure(num=idx, figsize=size)
    s = m.unpack_data(m.input_name, use_dout=False)[:1,:]
    r = m.unpack_data(m.output_name, use_dout=True)[:1,:]
    r2 = m.unpack_data("resp", use_dout=True)[:1,:]
    s2 = np.append(s, r, 0)
    s2 = np.append(s2, r2, 0)
    s2 = s2[:, s2[0, :].argsort()]
    bincount = np.min([100, s2.shape[1]])
    T = np.int(np.floor(s2.shape[1] / bincount))
    s2 = s2[:, 0:(T * bincount)]
    s2 = np.reshape(s2, [3, bincount, T])
    s2 = np.mean(s2, 2)
    s2 = np.squeeze(s2)

    plt.plot(s2[0, :], s2[1, :], 'k-')
    plt.plot(s2[0, :], s2[2, :], 'k.')
    plt.xlabel("Input ({0})".format(m.input_name))
    plt.ylabel("Output ({0})".format(m.output_name))
    # plt.title("{0}".format(m.name))


def scatter_smooth(m, idx=None, x_name=None, y_name=None, size=FIGSIZE):
    if idx:
        plt.figure(num=idx, figsize=size)
    if not x_name:
        x_name = m.output_name
    if not y_name:
        y_name = "resp"

    s = m.unpack_data(x_name, use_dout=True)
    r = m.unpack_data(y_name, use_dout=True)
    keepidx = np.isfinite(s[0, :]) * np.isfinite(r[0, :])
    s = s[0:1, keepidx]
    r = r[0:1, keepidx]

    s2 = np.append(s, r, 0)
    s2 = s2[:, s2[0, :].argsort()]
    bincount = np.min([100, s2.shape[1]])
    T = np.int(np.floor(s2.shape[1] / bincount))
    s2 = s2[:, 0:(T * bincount)]
    s2 = np.reshape(s2, [2, bincount, T])
    s2 = np.mean(s2, 2)
    s2 = np.squeeze(s2)
    plt.plot(s2[0, :], s2[1, :], 'k.')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    # m.parent_stack.meta['r_val']
    # plt.title("{0} (r_est={1:.3f}, r_val={2:.3f})".format(m.name,m.parent_stack.meta['r_est'],m.parent_stack.meta['r_val']))


def pred_act_scatter_smooth(m, idx=None, size=FIGSIZE):
    scatter_smooth(m, idx=idx, size=size, x_name=m.output_name, y_name="resp")


def state_act_scatter_smooth(m, idx=None, size=FIGSIZE):
    scatter_smooth(m, idx=idx, size=size, x_name=m.state_var, y_name="resp")
    if 'theta' in dir(m):
        t = "theta: " + " ".join(str(x) for x in m.theta)
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    if ymin == ymax:
        ymax = ymin + 1
    if xmin == xmax:
        xmax = xmin + 1
    # log.info("{0},{1} {2},{3}".format(xmin,xmax,ymin,ymax))
    plt.text(xmin + (xmax - xmin) / 50, ymax - (ymax - ymin) / 20, t,
             verticalalignment='top')


def pred_act_psth(m, size=FIGSIZE, idx=None):
    if idx:
        plt.figure(num=idx, figsize=size)
    out1 = m.d_out[m.parent_stack.plot_dataidx]
    s = out1[m.output_name][0, m.parent_stack.plot_stimidx, :]
    r = out1['resp'][0, m.parent_stack.plot_stimidx, :]
    fs = out1['respFs']
    tt = np.arange(0, len(r)) / fs
    pred, = plt.plot(tt, s, label='Predicted')
    act, = plt.plot(tt, r, 'r', label='Actual')
    plt.legend(handles=[pred, act])
    # plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time (s)')
    plt.ylabel('Firing rate (unitless)')


def pred_act_psth_smooth(m, size=FIGSIZE, idx=None):
    if idx:
        plt.figure(num=idx, figsize=size)
    out1 = m.d_out[m.parent_stack.plot_dataidx]
    s = out1[m.output_name][0, m.parent_stack.plot_stimidx, :]
    r = out1['resp'][0, m.parent_stack.plot_stimidx, :]

    box_pts = 20
    box = np.ones(box_pts) / box_pts
    s = np.convolve(s, box, mode='same')
    r = np.convolve(r, box, mode='same')

    pred, = plt.plot(s, label='Predicted')
    act, = plt.plot(r, 'r', label='Actual')
    plt.legend(handles=[pred, act])
    # plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time Step')
    plt.ylabel('Firing rate (unitless)')


def pred_act_psth_all(m, size=FIGSIZE, idx=None):
    if idx:
        plt.figure(num=idx, figsize=size)
    s = m.unpack_data(m.output_name, use_dout=True)
    r = m.unpack_data("resp", use_dout=True)
    s2 = np.append(s.transpose(), r.transpose(), 0)
    try:
        p = m.unpack_data("pupil", use_dout=True)
        s2 = np.append(s2, p.transpose(), 0)
        p_avail = True
    except BaseException:
        p_avail = False

    bincount = np.min([5000, s2.shape[1]])
    T = np.int(np.floor(s2.shape[1] / bincount))
    s2 = np.reshape(s2[:, 0:(T * bincount)], [3, T, bincount])
    s2 = np.mean(s2, 1)
    s2 = np.squeeze(s2)

    pred, = plt.plot(s2[0, :], label='Predicted')
    act, = plt.plot(s2[0, :], 'r', label='Actual')
    if p_avail:
        pup, = plt.plot(s2[0, :], 'g', label='Pupil')
        plt.legend(handles=[pred, act, pup])
    else:
        plt.legend(handles=[pred, act])

    # plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time Step')
    plt.ylabel('Firing rate (unitless)')


def pre_post_psth(m, size=FIGSIZE, idx=None):
    if idx:
        plt.figure(num=idx, figsize=size)
    in1 = m.d_in[m.parent_stack.plot_dataidx][m.input_name]
    out1 = m.d_out[m.parent_stack.plot_dataidx][m.output_name]
    if len(in1.shape) > 2:
        s1 = in1[0, m.parent_stack.plot_stimidx, :]
    else:
        s1 = in1[m.parent_stack.plot_stimidx, :]
    if len(out1.shape) > 2:
        s2 = out1[0, m.parent_stack.plot_stimidx, :]
    else:
        s2 = out1[m.parent_stack.plot_stimidx, :]

    pre, = plt.plot(s1, label='Pre-nonlinearity')
    post, = plt.plot(s2, 'r', label='Post-nonlinearity')
    plt.legend(handles=[pre, post])
    # plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time Step')
    plt.ylabel('Firing rate (unitless)')


def plot_stim_psth(m, idx=None, size=FIGSIZE):
    if idx:
        plt.figure(num=str(idx), figsize=size)
    out1 = m.d_out[m.parent_stack.plot_dataidx]
    # c=out1['repcount'][m.parent_stack.plot_stimidx]
    # h=out1[m.output_name][m.parent_stack.plot_stimidx].shape
    # scl=int(h[0]/c)
    s2 = out1[m.output_name][m.parent_stack.plot_stimidx, :]
    resp, = plt.plot(s2, 'r', label='Post-' + m.name)
    plt.legend(handles=[resp])
    #plt.title(m.name+': stim #'+str(m.parent_stack.plot_stimidx))
        
    
def plot_strf(m, idx=None, size=FIGSIZE):
    if 'bank_count' in dir(m) and m.bank_count>1:
        plot_strf_bank(m,idx,size)
        #plot_strf_bank(m,idx,size)
        return
 
    if idx:
        plt.figure(num=idx, figsize=size)
    h = m.coefs

    # if weight channels exist and dimensionality matches, generate a full STRF
    try:
        wcidx = nems.utilities.utils.find_modules(
            m.parent_stack, "filters.weight_channels")
        if len(wcidx) > 0 and m.parent_stack.modules[wcidx[0]].output_name == m.output_name:
            wcidx = wcidx[0]
        elif len(wcidx) > 1 and m.parent_stack.modules[wcidx[1]].output_name == m.output_name:
            wcidx = wcidx[1]
        else:
            wcidx = -1
    except BaseException:
        wcidx = -1

    if m.name == "filters.fir" and wcidx >= 0:
        # log.info(m.name)
        w = m.parent_stack.modules[wcidx].coefs
        if w.shape[0] == h.shape[0]:
            h = np.matmul(w.transpose(), h)

    #h=scipy.misc.imresize(h,(h.shape[0],h.shape[1]*3),'bilinear')
    mmax = np.max(np.abs(h.reshape(-1)))
    plt.imshow(h, aspect='auto', origin='lower',
               clim=[-mmax,mmax],cmap=plt.get_cmap('jet'), interpolation='none')
    cbar = plt.colorbar()
    cbar.set_label('gain')
    if m.name == "filters.fir":
        plt.xlabel('Latency')
        plt.ylabel('Channel')  # or kHz?
    elif m.name == "filters.weight_channels":
        plt.xlabel('Channel in')
        plt.ylabel('Channel out')  # or kHz?
    else:
        pass
        #plt.xlabel('Channel') #or kHz?
        #plt.ylabel('Latency')

def plot_strf_bank(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=idx,figsize=size)
        
    ax=plt.gca()
    b=ax.get_position()
    
    w=b.width/m.bank_count
    h=b.height
    ax.remove()
    
    axes = [plt.axes([b.min[0]+i*w, b.min[1], w*0.9,h]) for i in range(m.bank_count)]
    
    h=m.coefs
    stepsize=int(h.shape[0]/m.bank_count)
    
    # if weight channels exist and dimensionality matches, generate a full STRF
    try:
        wcidx=nems.utilities.utils.find_modules(m.parent_stack,"filters.weight_channels")
        if len(wcidx)>0 and m.parent_stack.modules[wcidx[0]].output_name==m.output_name:
            wcidx=wcidx[0]
        elif len(wcidx)>1 and m.parent_stack.modules[wcidx[1]].output_name==m.output_name:
            wcidx=wcidx[1]
        else:
            wcidx=-1
    except:
        wcidx=-1
        
    if m.name=="filters.fir" and wcidx>=0:
        w=m.parent_stack.modules[wcidx].coefs
        
        h_set=np.zeros([w.shape[1],h.shape[1],m.bank_count])
        for i in range(0,m.bank_count):
            idx=np.arange(stepsize*i,stepsize*(i+1))
            h_set[:,:,i]=np.matmul(w[idx,:].T, h[idx,:])
    else:
        h_set=np.reshape(h,[stepsize,m.bank_count,h.shape[1]])
        h_set=np.transpose(h_set,[0,2,1])
        
    h=np.reshape(h_set,[h_set.shape[0],-1],order='F')
    mmax=np.max(np.abs(h.reshape(-1)))

    for ii in range(m.bank_count):
        mmax = np.max(np.abs(h_set[:,:,ii].reshape(-1)))
        axes[ii].imshow(h_set[:,:,ii], aspect='auto', origin='lower',clim=[-mmax,mmax],cmap=plt.get_cmap('jet'), interpolation='none')
        if ii>0:
            axes[ii].tick_params(labelbottom='off',labelleft='off')
    #cbar = plt.gcf().colorbar(axes[-1])
    #cbar.set_label('gain')
    plt.sca(axes[0])
    if m.name=="filters.fir":
        plt.xlabel('Latency')
        plt.ylabel('Channel') #or kHz?
    elif m.name=="filters.weight_channels":
        plt.xlabel('Channel in')
        plt.ylabel('Channel out') #or kHz?
    else:
        pass
        #plt.xlabel('Channel') #or kHz?
        #plt.ylabel('Latency')
   

def non_plot(m):
    pass


def raster_data(data, pres, dura, posts, fr):
    s = data.shape
    pres = int(pres)
    dura = int(dura)
    posts = int(posts)
    xpre = np.zeros((s[2], pres * s[1]))
    ypre = np.zeros((s[2], pres * s[1]))
    xdur = np.zeros((s[2], dura * s[1]))
    ydur = np.zeros((s[2], dura * s[1]))
    xpost = np.zeros((s[2], posts * s[1]))
    ypost = np.zeros((s[2], posts * s[1]))
    for i in range(0, s[2]):
        spre = 0
        sdur = 0
        spost = 0
        for j in range(0, s[1]):
            ypre[i, spre:(spre + pres)] = (j + 1) * \
                                          np.clip(data[:pres, j, i], 0, 1)
            xpre[i, spre:(spre + pres)
            ] = np.divide(np.array(list(range(0, pres))), fr)
            ydur[i, sdur:(sdur + dura)] = (j + 1) * \
                                          np.clip(data[pres:(pres + dura), j, i], 0, 1)
            xdur[i, sdur:(
                sdur + dura)] = np.divide(np.array(list(range(pres, (pres + dura)))), fr)
            ypost[i, spost:(spost + posts)] = (j + 1) * \
                                              np.clip(data[(pres + dura):(pres + dura + posts), j, i], 0, 1)
            xpost[i, spost:(spost + posts)] = np.divide(
                np.array(list(range((pres + dura), (pres + dura + posts)))), fr)
            spre += pres
            sdur += dura
            spost += posts
    ypre[ypre == 0] = None
    ydur[ydur == 0] = None
    ypost[ypost == 0] = None
    return (xpre, ypre, xdur, ydur, xpost, ypost)


def raster_plot(m, idx=None, size=(12, 6)):
    """
    This function generates a raster plot of the data for the specified stimuli.
    It shows the spikes that occur during the actual trial in green, and the background
    spikes in grey.
    """
    resp = m.parent_stack.unresampled['resp']
    pre = m.parent_stack.unresampled['prestim']
    dur = m.parent_stack.unresampled['duration']
    post = m.parent_stack.unresampled['poststim']
    freq = m.parent_stack.unresampled['respFs']
    # log.info("{}/{}/{} fs {}".format(pre,dur,post,freq))
    total_bins = ((pre + dur + post) * freq).astype(int)
    # log.info(total_bins)
    if resp.shape[0] < total_bins:
        d = total_bins - resp.shape[0]
        s = list(resp.shape)
        s[0] = d
        resp = np.concatenate((resp, np.zeros(s)), axis=0)
    # log.info(resp.shape)
    reps = m.parent_stack.unresampled['repcount']
    ids = m.parent_stack.plot_stimidx
    r = reps.shape[0]
    prestim = float(pre) * freq
    duration = float(dur) * freq
    poststim = float(post) * freq
    if m.parent_stack.unresampled['pupil'] is not None:
        lis = []
        for i in range(0, r):
            lis.extend([i] * reps[i])
        stims = lis[ids]
    else:
        stims = ids
    xpre, ypre, xdur, ydur, xpost, ypost = raster_data(
        resp, prestim, duration, poststim, freq)
    if idx is not None:
        plt.figure(num=str(stims) + str(100), figsize=size)
    plt.scatter(xpre[stims], ypre[stims], color='0.5',
                s=(0.5 * np.pi) * 2, alpha=0.6)
    plt.scatter(xdur[stims], ydur[stims], color='g',
                s=(0.5 * np.pi) * 2, alpha=0.6)
    plt.scatter(xpost[stims], ypost[stims], color='0.5',
                s=(0.5 * np.pi) * 2, alpha=0.6)
    plt.ylabel('Trial')
    plt.xlabel('Time')
    plt.title('Stimulus #' + str(stims))


def sorted_raster(m, idx=None, size=FIGSIZE):
    """
    Creates a raster plot sorted by mean pupil diameter of a given trial
    """
    resp = m.parent_stack.unresampled['resp']
    pre = m.parent_stack.unresampled['prestim']
    dur = m.parent_stack.unresampled['duration']
    post = m.parent_stack.unresampled['poststim']
    freq = m.parent_stack.unresampled['respFs']
    reps = m.parent_stack.unresampled['repcount']
    r = reps.shape[0]
    prestim = float(pre) * freq
    duration = float(dur) * freq
    poststim = float(post) * freq
    pup = m.parent_stack.unresampled['pupil'][:,:,:,0]
    idi = m.parent_stack.plot_stimidx
    lis = []
    for i in range(0, r):
        lis.extend([i] * reps[i])
    ids = lis[idi]
    b = np.nanmean(pup[:, :, ids], axis=0)
    b = np.nan_to_num(b)
    bc = np.asarray(sorted(zip(b, range(0, len(b)))), dtype=int)
    bc = bc[:, 1]
    resp[:, :, ids] = resp[:, bc, ids]
    xpre, ypre, xdur, ydur, xpost, ypost = raster_data(
        resp, prestim, duration, poststim, freq)
    if idx is not None:
        plt.figure(num=str(ids) + str(100), figsize=size)
    plt.scatter(xpre[ids], ypre[ids], color='0.5',
                s=(0.5 * np.pi) * 2, alpha=0.6)
    plt.scatter(xdur[ids], ydur[ids], color='g',
                s=(0.5 * np.pi) * 2, alpha=0.6)
    plt.scatter(xpost[ids], ypost[ids], color='0.5',
                s=(0.5 * np.pi) * 2, alpha=0.6)
    plt.ylabel('Trial')
    plt.xlabel('Time')
    plt.title('Sorted by Pupil: Stimulus #' + str(ids))


def plot_ssa_idx(m, idx=None, size=FIGSIZE,
                 figure=None, outer=None, error=False):
    '''
    specific plotting function for the ssa_index module, essentially overlayed PSTHs for each tone type.
    '''
    if idx:
        figure = plt.figure(num=idx, figsize=size)

    if isinstance(outer, gridspec.SubplotSpec):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=outer)  # ,wspace=0.1, hspace=0.1)
    elif figure is None:
        figure = plt.figure(num=idx, figsize=size)
        inner = gridspec.GridSpec(1, 2)
    else:
        raise (
            '"outer" has to be an instance of gridspec.GridSpecFromSubplotSpec or None (default)')

    has_pred = m.has_pred

    folded_resp = m.folded_resp[m.parent_stack.plot_stimidx]

    if has_pred:
        folded_pred = m.folded_pred[m.parent_stack.plot_stimidx]

    # smooths the data, right now deactivated since the import already
    # downsamples.
    smooth = False
    if smooth:
        box_pts = 20
    else:
        box_pts = 1

    box = np.ones(box_pts) / box_pts

    # Calculates traces and errors for each tone type of response
    resp_dict = {key: (np.convolve(np.nanmean(value, axis=0), box, mode='same'))
                 for key, value in folded_resp.items()}

    if error:
        resp_err_dict = {key: (np.convolve(np.nanstd(value, axis=0), box, mode='same'))
                         for key, value in folded_resp.items()}

    # calculates traces and errors for each tone type of predictions, or whole
    # cell values if no pred.

    if has_pred:
        pred_dict = {key: (np.convolve(np.nanmean(value, axis=0), box, mode='same'))
                     for key, value in folded_pred.items()}

        if error:
            pred_err_dict = {key: (np.convolve(np.nanstd(value, axis=0), box, mode='same'))
                             for key, value in folded_pred.items()}

    else:
        tone_type = ['Std', 'Dev']

        cell_act = {key: (
            np.convolve(
                np.nanmean(np.concatenate(
                    [val for k, val in folded_resp.items() if k[-3:] == key], axis=0), axis=0),
                box, mode='same')) for key in tone_type}
        if error:
            cell_err = {key: (
                np.convolve(
                    np.nanstd(np.concatenate(
                        [val for k, val in folded_resp.items() if k[-3:] == key], axis=0), axis=0),
                    box, mode='same')) for key in tone_type}

    # plotting parameters: keys = Tone types to be ploted; colors = color of line, correspond with stream;
    # Lines = type of line, correspond with standard or deviant .
    keys = ['stream0Std', 'stream0Dev', 'stream1Std', 'stream1Dev']
    colors = ['C0', 'C0', 'C1', 'C1']
    lines = ['-', ':', '-', ':']

    axes = [plt.Subplot(figure, ax) for ax in inner]
    x_ax = resp_dict['stream0Std'].shape[0]

    # First part: plot of cell response by tone type.

    for k, c, l in zip(keys, colors, lines):
        axes[0].plot(resp_dict[k], color=c, linestyle=l, label=k)
        if error:
            axes[0].fill_between(range(x_ax), resp_dict[k] - resp_err_dict[k], resp_dict[k] + resp_err_dict[k],
                                 facecolor=c, alpha=0.2)

    axes[0].axvline(x_ax / 3, color='black')
    axes[0].axvline((x_ax / 3) * 2, color='black')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Firing Rate')
    axes[0].legend(loc='upper left', fontsize='xx-small')

    # second part: plot of predicted cell activity by tone type.

    if has_pred:

        for k, c, l in zip(keys, colors, lines):
            axes[1].plot(pred_dict[k], color=c, linestyle=l, label=k)
            if error:
                axes[1].fill_between(range(x_ax), pred_dict[k] - pred_err_dict[k], pred_dict[k] + pred_err_dict[k],
                                     facecolor=c, alpha=0.2)
    else:
        lines = ['-', ':']

        for k, l in zip(tone_type, lines):
            axes[1].plot(cell_act[k], color='black', linestyle=l, label=k)
            if error:
                axes[1].fill_between(range(x_ax), cell_act[k] - cell_err[k], cell_act[k] + cell_err[k],
                                     facecolor='gray', alpha=0.2)

    axes[1].axvline(x_ax / 3, color='black')
    axes[1].axvline((x_ax / 3) * 2, color='black')
    axes[1].set_xlabel('Time Step')
    axes[1].set_ylabel('Firing Rate')
    axes[1].legend(loc='upper left', fontsize='xx-small')

    for ax in axes:
        figure.add_subplot(ax)

    # sets the y axis so they are shared
    axes[1].set_ylim(axes[0].get_ylim())


def plot_stp(m, idx=None, size=FIGSIZE, figure=None, outer=None, error=False):
    '''
    specific plotting function for the stp module, creates a standarized two tone stimulus and evaluates
    it with the fitted stp parameters i.e. Tau and U. Plots the response and displays Tau, U and SI

    '''
    if idx:
        figure = plt.figure(num=idx, figsize=size)

    if isinstance(outer, gridspec.SubplotSpec):
        inner = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer)  # ,wspace=0.1, hspace=0.1)
    elif outer == None:
        figure = plt.figure(num=idx, figsize=size)
        inner = gridspec.GridSpec(1, 2)
    else:
        raise (
            '"outer" has to be an instance of gridspec.GridSpecFromSubplotSpec or None (default)')

    # creates a two tone paradigm with varying interval
    pair_cout = 5  # number of pairs to be analized
    toneLen = 0.100  # tone length in seconds
    isi = toneLen  # in seconds
    flanks = toneLen * 2  # silences flanking the stimulus  in seconds

    amplitude = np.nanmax(
        m.d_in[0]['stim'])  # TODO change to have the same amplitude as the stated in the experiment parameters
    sf = m.d_in[0]['stimFs']

    stims = list()
    for ii in range(0, pair_cout + 1):
        pair_pulse = np.zeros([2, np.int(pair_cout * sf * (toneLen + isi) + sf * flanks * 2)])
        pair_pulse[:, int(sf * flanks): np.int(sf * (flanks + toneLen))] = amplitude  # defines the first tone
        if ii == 0:
            # first trial only has the first tone
            stims.append(pair_pulse)
        else:
            start = np.int(sf * (flanks + ii * (toneLen + isi)))
            end = np.int(sf * (flanks + ii * (toneLen + isi) + toneLen))
            pair_pulse[:, start: end] = amplitude  # defines the second tone
            stims.append(pair_pulse)

    stims = np.asarray(stims)
    stims = stims.swapaxes(0, 1)  # dim0: streams, dim1: trials, dim2: time

    # Here just paste the stp module my_eval, dont know how to be more elegant.
    s = stims.shape
    tstim = (stims > 0) * stims
    Y = np.zeros([0, s[1], s[2]])
    di = np.ones(s)
    for j in range(0, m.num_channels):  # not sure what channels are these

        ui = np.absolute(m.u[:, j])  # force only depression, no facilitation
        # convert tau units from sec to bins
        taui = np.absolute(m.tau[:, j]) * m.d_in[0]['fs']

        # go through each stimulus channel
        for i in range(0, s[0]):

            for tt in range(1, s[2]):
                td = di[i, :, tt - 1]  # previous time bin depression
                if ui[i] > 0:
                    delta = (1 - td) / taui[i] - ui[i] * td * tstim[i, :, tt - 1]
                    td = td + delta
                    td[td < 0] = 0
                else:
                    delta = (1 - td) / taui[i] - ui[i] * td * tstim[i, :, tt - 1]
                    td = td + delta
                    td[td < 1] = 1
                di[i, :, tt] = td

        Y = np.append(Y, di * stims, 0)

    # does the plotting

    axes = [plt.Subplot(figure, ax) for ax in inner]
    # for test purposes
    '''
    fig, axes = plt.subplots(2,1, sharex=True, sharey=True)
    axes = np.ravel(axes)
    '''

    xtime = np.arange(s[2]) / sf

    for ii in range(s[1]):
        plot_offset = 1.5
        # uses legend for Tau and U notation once
        if ii == 0:
            axes[0].plot(xtime, Y[0, ii, :], color='C0',
                         label='Tau = {:.3f}, U = {:.3f}'.format(m.tau[0][0], m.u[0][0]))
            axes[1].plot(xtime, Y[1, ii, :], color='C1',
                         label='Tau = {:.3f}, U = {:.3f}'.format(m.tau[1][0], m.u[1][0]))
        else:
            axes[0].plot(xtime, Y[0, ii, :], color='C0', alpha=0.5)
            axes[1].plot(xtime, Y[1, ii, :], color='C1', alpha=0.5)
    for ax in axes:
        ax.set_xlabel('seconds')
        ax.set_ylabel('activity, AU')
        ax.legend()
        figure.add_subplot(ax)


#
# Other support functions
#

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
        # log.info('start loop 1')
        # log.info(len(stack.data[k]))
        for n in range(0, len(stack.data[k])):
            # log.info('start loop 2')
            if stack.data[k][n]['est'] is False:
                # log.info('concatenating')
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
    s_indices = resamp < thresh
    resamp[s_indices] = 0
    return resamp


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

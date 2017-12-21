#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules for importing data into the stack object


Created on Fri Aug  4 13:14:24 2017

@author: shofer
"""

import logging
log = logging.getLogger(__name__)

from nems.modules.base import nems_module
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

import nems.utilities.utils
import nems.utilities.plot
import nems.utilities.io


class load_mat(nems_module):
    """
    Loads a MATLAB data file (.mat file) containing several "structs" which have
    data for an individual cell.

    Inputs:
        fs: frequency to resample stimulus, response, and pupil data.
        avg_resp: average all trials in the response raster and place in
                the output dictionary as 'resp'. Usually used when pupil
                effect are being considered, and will generally allow for
                better fitting.
        est_files: MATLAB data files to load.

    Returns: Data from this file is loaded into the stack.data as a list of dictionaries
    with keywords:
        'resp': response raster for each type of stimulus
        'stim': stimuli spectrograms that correspond to response
        'respFs': sampling frequency of response raster
        'stimFs':sampling frequency of stimuli spectrograms
        'stimparam': details on types of stimuli used
        'isolation': isolation of recorded cells (?)
        'prestim': length of silence before stimulus begins
        'poststim': length of silence after stimulus ends
        'duration': length of simulus
        'pupil': continuous pupil diameter measurements
        'est': flag for estimation/validation data
        'repcount': how many trials of each stimulus are present
        'replist': a list containing the number of each stimulus the number of
                times it was played. E.g., if we have stimulus 1 that was played
                3 times and stimulus 2 that was played 2 times, replist would
                be [1,1,1,2,2].
    """
    name = 'loaders.load_mat'
    user_editable_fields = ['output_name', 'est_files', 'fs', 'avg_resp']
    plot_fns = [nems.utilities.plot.plot_spectrogram,
                nems.utilities.plot.plot_spectrogram]
    est_files = []
    fs = 100
    avg_resp = True

    def my_init(self, est_files=[], fs=100, avg_resp=True, filestate=False):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.est_files = est_files.copy()
        self.fs = fs
        self.avg_resp = avg_resp
        self.parent_stack.avg_resp = avg_resp
        self.filestate = filestate
        self.auto_plot = False

    def evaluate(self, **kwargs):

        # intialize by deleting any existing entries in self.d_out
        del self.d_out[:]
#        for i, d in enumerate(self.d_in):
#            self.d_out.append(d.copy())

        # load contents of Matlab data file and save in d_out list
        for f in self.est_files:
            matdata = nems.utilities.io.get_mat_file(f)

            # go through each entry in structure array 'data'
            for s in matdata['data'][0]:

                data = {}
                if 'stimids' in s.dtype.names:
                    # new format: stimulus events logged in stimids and
                    # pulled from stim matrix or from a separate file
                    tstim = s['stim']
                    stimids = s['stimids']
                    stimtrials = s['stimtrials']
                    stimtimes = np.double(s['stimtimes'])
                    stimshape = tstim.shape
                    respshape = s['resp_raster'].shape
                    chancount = stimshape[0]
                    stimbins = np.round(stimtimes * np.double(s['stimfs']))
                    stim = np.zeros([chancount, respshape[0], respshape[2]])
                    eventcount = len(stimtimes)
                    for ii in range(0, eventcount):
                        startbin = np.int(stimbins[ii])
                        stopbin = startbin + stimshape[1]
                        if stimids[ii] < stimshape[2] and stopbin <= respshape[0]:
                            stim[:, startbin:stopbin, stimtrials[ii] -
                                 1] = tstim[:, :, stimids[ii] - 1]
                    data['stim'] = stim

                else:
                    # old format, stimulus saved as raster aligned with spikes
                    data['stim'] = s['stim']

                try:
                    data['resp'] = s['resp_raster']
                    data['respFs'] = s['respfs'][0][0]
                    data['stimFs'] = s['stimfs'][0][0]
                    data['stimparam'] = [str(''.join(letter))
                                         for letter in s['fn_param']]
                    data['isolation'] = s['isolation']
                    data['prestim'] = s['tags'][0]['PreStimSilence'][0][0][0]
                    data['poststim'] = s['tags'][0]['PostStimSilence'][0][0][0]
                    data['duration'] = s['tags'][0]['Duration'][0][0][0]
                except BaseException:
                    log.info("load_mat: alternative load. does this ever execute?")
                    data = scipy.io.loadmat(f, chars_as_strings=True)
                    data['raw_stim'] = data['stim'].copy()
                    data['raw_resp'] = data['resp'].copy()
                try:
                    data['pupil'] = s['pupil'] / 100
                except BaseException:
                    data['pupil'] = None
                try:
                    data['state'] = s['state']
                except BaseException:
                    data['state'] = None
                # data['tags']=s.get('tags',None)

                try:
                    if s['estfile']:
                        data['est'] = True
                    else:
                        data['est'] = False
                except ValueError:
                    pass
                    #log.info("Est/val conditions not flagged in datafile")
                try:
                    data['filestate'] = s['filestate'][0][0]
                except BaseException:
                    data['filestate'] = 0

                # deal with extra dimensions in RDT data
                if data['stim'].ndim > 3:
                    data['stim1'] = data['stim'][:, :, :, 1]
                    data['stim2'] = data['stim'][:, :, :, 2]
                    data['stim'] = data['stim'][:, :, :, 0]
                    stimvars = ['stim', 'stim1', 'stim2']
                else:
                    stimvars = ['stim']

                # resample if necessary
                data['fs'] = self.fs
                noise_thresh = 0.05
                stim_resamp_factor = int(data['stimFs'] / data['fs'])
                resp_resamp_factor = int(data['respFs'] / data['fs'])

                self.parent_stack.unresampled = {'resp': data['resp'], 'respFs': data['respFs'], 'duration': data['duration'],
                                                 'poststim': data['poststim'], 'prestim': data['prestim'], 'pupil': data['pupil']}

                for sname in stimvars:
                    # reshape stimulus to be channel X time
                    data[sname] = np.transpose(data[sname], (0, 2, 1))

                    if stim_resamp_factor in np.arange(0, 10):
                        data[sname] = nems.utilities.utils.bin_resamp(
                            data[sname], stim_resamp_factor, ax=2)

                    elif stim_resamp_factor != 1:
                        data[sname] = nems.utilities.utils.thresh_resamp(
                            data[sname], stim_resamp_factor, thresh=noise_thresh, ax=2)

                # resp time (axis 0) should be resampled to match stim time
                # (axis 1)

                # Changed resample to decimate w/ 'fir' and threshold, as it produces less ringing when downsampling
                #-njs June 16, 2017
                if resp_resamp_factor in np.arange(0, 10):
                    log.info("resp bin resamp factor {0}".format(
                        resp_resamp_factor))
                    data['resp'] = nems.utilities.utils.bin_resamp(
                        data['resp'], resp_resamp_factor, ax=0)
                    if data['pupil'] is not None:
                        data['pupil'] = nems.utilities.utils.bin_resamp(
                            data['pupil'], resp_resamp_factor, ax=0)
                        # save raw pupil-- may be somehow transposed
                        # differently than resp_raw
                        data['pupil_raw'] = data['pupil'].copy()

                elif resp_resamp_factor != 1:
                    data['resp'] = nems.utilities.utils.thresh_resamp(
                        data['resp'], resp_resamp_factor, thresh=noise_thresh)
                    if data['pupil'] is not None:
                        data['pupil'] = nems.utilities.utils.thresh_resamp(
                            data['pupil'], resp_resamp_factor, thresh=noise_thresh)
                        # save raw pupil-- may be somehow transposed
                        # differently than resp_raw
                        data['pupil_raw'] = data['pupil'].copy()

                # fund number of reps of each stimulus
                data['repcount'] = np.sum(
                    np.isfinite(data['resp'][0, :, :]), axis=0)
                self.parent_stack.unresampled['repcount'] = data['repcount']

                # average across trials
                # TODO - why does this execute(and produce a warning?)
                if data['resp'].shape[1] > 1:
                    data['avgresp'] = np.nanmean(data['resp'], axis=1)
                else:
                    data['avgresp'] = np.squeeze(data['resp'], axis=1)

                data['avgresp'] = np.transpose(data['avgresp'], (1, 0))

                if self.avg_resp is True:
                    data['resp_raw'] = data['resp'].copy()
                    data['resp'] = data['avgresp']
                else:
                    data['stim'], data['resp'], data['pupil'], data['replist'] = nems.utilities.utils.stretch_trials(
                        data)
                    data['resp_raw'] = data['resp']

                # new: add extra first dimension to resp/pupil (and eventually pred)
                # resp,pupil,state,pred now channel X stim/trial X time
                data['resp'] = data['resp'][np.newaxis, :, :]

                data['behavior_condition'] = np.ones(
                    data['resp'].shape) * (data['filestate'] > 0)
                data['behavior_condition'][np.isnan(data['resp'])] = np.nan

                if data['pupil'] is not None:
                    if data['pupil'].ndim == 3:
                        data['pupil'] = np.transpose(data['pupil'], (1, 2, 0))
                        if self.avg_resp is True:
                            data['state'] = np.concatenate((np.mean(data['pupil'], 0)[np.newaxis, :, :],
                                                            data['behavior_condition']), 0)
                        else:
                            data['state'] = data['behavior_condition']

                    elif data['pupil'].ndim == 2:
                        data['pupil'] = data['pupil'][np.newaxis, :, :]
                        # add file state as second dimension to pupil
                        data['state'] = np.concatenate((data['pupil'],
                                                        data['behavior_condition']), axis=0)

                else:
                    data['state'] = data['behavior_condition']

                # append contents of file to data, assuming data is a dictionary
                # with entries stim, resp, etc...
                #log.info('load_mat: appending {0} to d_out stack'.format(f))
                self.d_out.append(data)

        # Raises error if d_out is an empty list
        if not self.d_out:
            raise IndexError('loader module d_out is empty')


class load_gen(nems_module):
    """
    load_gen : general-purpose loading wrapper. currently only supports load_ecog
    """
    name = 'loaders.load_gen'
    user_editable_fields = ['output_name',
                            'stimfile', 'respfile', 'fs', 'avg_resp', 'resp_channels']
    plot_fns = [nems.utilities.plot.plot_spectrogram,
                nems.utilities.plot.raster_plot]
    stimfile = None
    respfile = None
    fs = 100
    avg_resp = True
    resp_channels=[0]
    
    def my_init(self, stimfile=None, respfile=None, fs=100,
                avg_resp=True, load_fun='load_ecog', resp_channels=[0]):
        self.stimfile = stimfile
        self.respfile = respfile
        self.fs = fs
        self.avg_resp = avg_resp
        self.auto_plot = False
        self.resp_channels = resp_channels
        self.load_fun = load_fun
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
        if self.load_fun=='load_ecog':
            self.d_out[0] = nems.utilities.io.load_ecog(
                    stack=self.parent_stack, fs=self.fs, 
                    avg_resp=self.avg_resp, respfile=self.respfile,
                    stimfile=self.stimfile, resp_channels=self.resp_channels)
        elif self.load_fun=='load_factor':
            self.d_out[0] = nems.utilities.io.load_factor(
                    stack=self.parent_stack, fs=self.fs, 
                    avg_resp=self.avg_resp, respfile=self.respfile,
                    stimfile=self.stimfile, resp_channels=self.resp_channels)
        else:
            raise ValueError('Unsupported load_fun')


class dummy_data(nems_module):
    """
    dummy_data - generate some very dumb test data without loading any files.
    Maybe deprecated?
    """
    name = 'loaders.dummy_data'
    user_editable_fields = ['output_name', 'data_len', 'fs']
    plot_fns = [nems.utilities.plot.plot_spectrogram]
    data_len = 100
    fs = 100

    def my_init(self, data_len=100, fs=100):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.data_len = data_len
        self.fs = fs

    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        self.d_out[0][self.output_name] = np.zeros([12, 2, self.data_len])
        self.d_out[0][self.output_name][0, 0, 10:19] = 1
        self.d_out[0][self.output_name][0, 0, 30:49] = 1
        self.d_out[0]['resp'] = self.d_out[0]['stim'][0, :, :] * 2 + 1
        self.d_out[0]['repcount'] = np.sum(
            np.isnan(self.d_out[0]['resp']) == False, axis=0)

class load_signals(load_mat):

    def my_init(self, signals=[]):
        self.signals = signals

    def evaluate(self, **kwargs):
        del self.d_out[:]

        stims = [s for s in self.signals if 'stim' in s.name]
        resps = [s for s in self.signals if 'resp' in s.name]
        pupils = [s for s in self.signals if 'pupil' in s.name]

        # TODO: Think of more intelligent ways of combining multiple signals
        # At the moment, it dies if there are >1 signals of these types:
        #    stim
        #    resp
        #    pupil
        # In the future, we could make this smarter (concatenate automatically?)

        if (len(stims) > 1 or len(resps) > 1 or len(pupils) > 1):
            raise ValueError(["Cannot determine stim/resp name; if you want "
                              "to combine multiple stims or responses you "
                              "need to do that manually first. "])

        stim = stims[0]
        resp = resps[0]
        pupil = pupils[0] if pupils else None

        data = {}
        data['stim'] = stim.as_old_matlab_format()
        data['resp'] = resp.as_old_matlab_format()
        data['stimFs'] = stim.fs
        data['respFs'] = resp.fs
        data['isolation'] = resp.meta['isolation']
        data['prestim'] = stim.meta['prestim']
        data['poststim'] = stim.meta['poststim']
        data['duration'] = stim.meta['duration']
        data['pupil'] = pupil.as_old_matlab_format()
        data['filestate'] = 0

        # DELETED: Resampling stuff.
        # fund number of reps of each stimulus
        # data['repcount'] = np.sum(np.isfinite(data['resp'][0,:,:]),axis=0)
        # data['avg_resp'] = ...

        #data['behavior_condition'] = np.ones(data['resp'].shape)*(data['filestate']>0)
        #data['behavior_condition'][np.isnan(data['resp'])]=np.nan

        log.info('Exiting load_mat_hacked ')
        self.d_out.append(data)

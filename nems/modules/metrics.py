#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules for computing scores/ assessing model performance

Created on Fri Aug  4 13:44:42 2017

s"""
from nems.modules.base import nems_module
import nems.utilities.utils
import nems.utilities.plot

import numpy as np
import scipy.stats as spstats

class mean_square_error(nems_module):

    name='metrics.mean_square_error'
    user_editable_fields=['input1','input2','norm','shrink']
    plot_fns=[nems.utilities.plot.pred_act_psth,nems.utilities.plot.pred_act_psth_smooth,nems.utilities.plot.pred_act_scatter]
    input1='pred'
    input2='resp'
    norm=True
    shrink=0
    mse_est=np.ones([1,1])
    mse_val=np.ones([1,1])

    def my_init(self, input1='pred',input2='resp',norm=True,shrink=0):
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.input1=input1
        self.input2=input2
        self.norm=norm
        self.shrink=shrink
        self.do_trial_plot=self.plot_fns[1]

    def evaluate(self,nest=0):
        if nest==0:
            del self.d_out[:]
            for i, d in enumerate(self.d_in):
                self.d_out.append(d.copy())
            
        X1=self.unpack_data(self.input1,est=True)
        X2=self.unpack_data(self.input2,est=True)
        keepidx=np.isfinite(X1) * np.isfinite(X2)
        X1=X1[keepidx]
        X2=X2[keepidx]
        if self.shrink:
            bounds=np.round(np.linspace(0,len(X1)+1,11)).astype(int)
            E=np.zeros([10,1])
            #P=np.mean(np.square(X2))
            for ii in range(0,10):
                E[ii]=np.mean(np.square(X1[bounds[ii]:bounds[ii+1]]-X2[bounds[ii]:bounds[ii+1]]))/np.mean(np.square(X2[bounds[ii]:bounds[ii+1]]))
            #E=E/P
            
            mE=E.mean()
            sE=E.std()
            
            if self.parent_stack.valmode:
                print(E)
                print(mE)
                print(sE)
                print("MSE shrink: {0}".format(self.shrink))
                
            if mE<1:
                # apply shrinkage filter to 1-E with factors self.shrink
                mse=1-nems.utilities.utils.shrinkage(1-mE,sE,self.shrink)
            else:
                mse=mE

        else:
            E=np.sum(np.square(X1-X2))
            P=np.sum(X2*X2)
            N=X1.size

#            E=np.zeros([1,1])
#            P=np.zeros([1,1])
#            N=0
#            for f in self.d_out:
#                #try:
#                E+=np.sum(np.square(f[self.input1]-f[self.input2]))
#                P+=np.sum(np.square(f[self.input2]))
#                #except TypeError:
#                    #print('error eval')
#                    #nems.utilities.utils.concatenate_helper(self.parent_stack)
#                    #E+=np.sum(np.square(f[self.input1]-f[self.input2]))
#                    #P+=np.sum(np.square(f[self.input2]))
#                N+=f[self.input2].size

            if self.norm:
                if P>0:
                    mse=E/P
                else:
                    mse=1
            else:
                mse=E/N

        self.mse_est=mse
        self.parent_stack.meta['mse_est']=[mse]
                
        if self.parent_stack.valmode:
            
            X1=self.unpack_data(self.input1,est=False)            
            X2=self.unpack_data(self.input2,est=False)
            keepidx=np.isfinite(X1) * np.isfinite(X2)
            X1=X1[keepidx]
            X2=X2[keepidx]
            E=np.sum(np.square(X1-X2))
            P=np.sum(X2*X2)
            N=X1.size

            if self.norm:
                if P>0:
                    mse=E/P
                else:
                    mse=1    
            else:
                mse=E/N
            self.mse_val=mse
            self.parent_stack.meta['mse_val']=[mse]

        return mse

    def error(self, est=True):
        if est:
            return self.mse_est
        else:
            # placeholder for something that can distinguish between est and val
            return self.mse_val


class likelihood_poisson(nems_module):

    name='metrics.likelihood_poisson'
    user_editable_fields=['input1','input2','shrink']
    plot_fns=[nems.utilities.plot.pred_act_psth,nems.utilities.plot.pred_act_psth_smooth,nems.utilities.plot.pred_act_scatter]
    input1='pred'
    input2='resp'
    norm=True
    shrink=0
    ll_est=np.zeros([1,1])
    ll_val=np.zeros([1,1])

    def my_init(self, input1='pred',input2='resp',shrink=False):
        self.input1=input1
        self.input2=input2
        self.shrink=shrink
        self.do_trial_plot=self.plot_fns[1]

    def evaluate(self,nest=0):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        X1=self.unpack_data(self.input1,est=True)
        X2=self.unpack_data(self.input2,est=True)
        keepidx=np.isfinite(X1) * np.isfinite(X2)
        X1=X1[keepidx]
        X2=X2[keepidx]
        X1[X1<0.00001]=0.00001

        ll_est=np.mean(X2 * np.log(X1) - X1) / np.mean(X2)

        self.ll_est=ll_est
        self.parent_stack.meta['ll_est']=[ll_est]

        #ee(bb)= -nanmean(r(bbidx).*log(p(bbidx)) - p(bbidx))./(d+(d==0));

        X1=self.unpack_data(self.input1,est=False)

        if X1.size:
            X2=self.unpack_data(self.input2,est=False)
            keepidx=np.isfinite(X1) * np.isfinite(X2)
            X1=X1[keepidx]
            X2=X2[keepidx]
            X1[X1<0.00001]=0.00001

            ll_val=-np.mean(X2 * np.log(X1) - X1) / np.mean(X2)

            self.ll_val=ll_val
            self.parent_stack.meta['ll_val']=[ll_val]

            return [ll_val]
        else:
            return [ll_est]


    def error(self, est=True):
        if est:
            return self.ll_est
        else:
            # placeholder for something that can distinguish between est and val
            return self.ll_val




class pseudo_huber_error(nems_module):
    """
    Pseudo-huber "error" to use with fitter cost functions. This is more robust to
    ouliers than simple mean square error. Approximates L1 error at large
    values of error, and MSE at low error values. Has the additional benefit (unlike L1)
    of being convex and differentiable at all places.
    
    Pseudo-huber equation taken from Hartley & Zimmerman, "Multiple View Geometry
    in Computer Vision," (Cambridge University Press, 2003), p619
    
    C(delta)=2(b^2)(sqrt(1+(delta/b)^2)-1)
    
    b mediates the value of error at which the the error is penalized linearly or quadratically.
    Note that setting b=1 is the soft l1 loss
    
    @author: shofer, June 30 2017
    """
    #I think this is working (but I'm not positive). When fitting with a pseudo-huber
    #cost function, the fitter tends to ignore areas of high spike rates, but does
    #a good job finding the mean spike rate at different times during a stimulus.
    #This makes sense in that the huber error penalizes outliers, and could be
    #potentially useful, depending on what is being fit? --njs, June 30 2017


    name='metrics.pseudo_huber_error'
    user_editable_fields=['input1','input2','b']
    plot_fns=[nems.utilities.plot.pred_act_psth,nems.utilities.plot.pred_act_scatter]
    input1='pred'
    input2='resp'
    b=0.9 #sets the value of error where fall-off goes from linear to quadratic\
    huber_est=np.ones([1,1])
    huber_val=np.ones([1,1])

    def my_init(self, input1='pred',input2='resp',b=0.9):
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.input1=input1
        self.input2=input2
        self.b=b
        self.do_trial_plot=self.plot_fns[1]

    def evaluate(self,nest=0):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        for f in self.d_out:
            delta=np.divide(np.sum(f[self.input1]-f[self.input2],axis=1),np.sum(f[self.input2],axis=1))
            C=np.sum(2*np.square(self.b)*(np.sqrt(1+np.square(np.divide(delta,self.b)))-1))
            C=np.array([C])
        self.huber_est=C

    def error(self,est=True):
        if est is True:
            return(self.huber_est)
        else:
            return(self.huber_val)

class correlation(nems_module):

    name='metrics.correlation'
    user_editable_fields=['input1','input2','norm']
    plot_fns=[nems.utilities.plot.pred_act_psth, nems.utilities.plot.pred_act_scatter, nems.utilities.plot.pred_act_scatter_smooth]
    input1='pred'
    input2='resp'
    r_est=np.ones([1,1])
    r_val=np.ones([1,1])

    def my_init(self, input1='pred',input2='resp',norm=True):
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.input1=input1
        self.input2=input2
        self.do_plot=self.plot_fns[1]

    def evaluate(self,**kwargs):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        X1=self.unpack_data(self.input1,est=True)
        X2=self.unpack_data(self.input2,est=True)
        keepidx=np.isfinite(X1) * np.isfinite(X2)
        X1=X1[keepidx]
        X2=X2[keepidx]
        if not X1.sum() or not X2.sum():
            r_est=0
        else:
            r_est,p=spstats.pearsonr(X1,X2)
        self.r_est=r_est
        self.parent_stack.meta['r_est']=[r_est]

        X1=self.unpack_data(self.input1,est=False)
        if X1.size:
            X2=self.unpack_data(self.input2,est=False)
            keepidx=np.isfinite(X1) * np.isfinite(X2)
            X1=X1[keepidx]
            X2=X2[keepidx]
            if not X1.sum() or not X2.sum():
                r_val=0
            else:
                r_val,p=spstats.pearsonr(X1,X2)
            self.r_val=r_val
            self.parent_stack.meta['r_val']=[r_val]

            # if running validation test, also measure r_floor
            rf=np.zeros([1000,1])
            for rr in range(0,len(rf)):
                n1=(np.random.rand(500,1)*len(X1)).astype(int)
                n2=(np.random.rand(500,1)*len(X2)).astype(int)
                rf[rr],p=spstats.pearsonr(X1[n1],X2[n2])
            rf=np.sort(rf[np.isfinite(rf)],0)
            if len(rf):
                self.parent_stack.meta['r_floor']=[rf[np.int(len(rf)*0.95)]]
            else:
                self.parent_stack.meta['r_floor']=0

            return [r_val]
        else:
            return [r_est]

class ssa_index(nems_module):
    '''
    SSA index (SI) calculations as stated by Ulanovsky et al., 2003. The module take a in stimulus envelope input
    with 3 dimensions corresponding to stream (tone 1 or tone 2), trial and time; and a response input lacking
    the first dimension.

    Using the envelope defines for each trial which tone is being standard, deviant and onset, and then precedes to
    cut the response to such tones and pool them in 6 bins (3 tone natures times to streams)

    for each pool, all tone responses are averaged and then the average is integrated from the onset of the tone to
    twice the lenght of the tone (to include any offset responses).

    for each stream (tone 1 or 2) the SI is calculated as:

        (tone n deviant - tone n standard) / (tone n devian + tone n standard)

    SI can also be calculated for the whole cell in a tone "independent" way:

        (tone1 deviant + tone2 deviant - tone1 standard - tone2 standard) /
        (tone1 deviant + tone2 deviant + tone1 standard + tone2 standard)

    Aditionally given that ssa is a consequence of stp, which is related to tone interval. for any given tone the time
    since the preceding tone is also extracted and organized in a dictionary of the same shape
    '''

    name = 'metrics.ssa_index'
    user_editable_fields = ['input1', 'input2', 'baseline', 'window']
    plot_fns = [nems.utilities.plot.plot_ssa_idx, nems.utilities.plot.plot_ssa_timing]
    input1 = 'stim'
    input2 = 'resp'
    window = 'start'

    resp_SI = dict()
    pred_SI = dict()

    # for raster and PSTH plotting
    folded_resp = list()
    folded_pred = list()

    # for tone timing plot and stp calculation
    intervals = list()
    resp_tone_act = list()
    pred_tone_act = list()

    def my_init(self, input1='stim', input2='resp', window='start', z_score = 'spont'):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.input1 = input1
        self.input2 = input2
        self.window = window
        self.do_plot = self.plot_fns[0]
        self.do_trial_plot = self.plot_fns[0]
        self.has_pred = False
        self.z_score = z_score

    def evaluate(self, **kwargs):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        intervals = list()

        resp_SI = list()
        folded_resp = list()
        resp_tone_act = list()
        cell_resp_spont = list()

        pred_SI = list()
        folded_pred = list()
        pred_tone_act = list()
        cell_pred_spont = list()

        out_SI_dicts = list()
        out_act_dicts = list()

        # if validation is active picks only estimation blocks for SSA Index calculation. Validation subsets can be
        # inconveniently short, this leads to lack of deviants and standards for one or other streams, preventing any
        # calculation of ssa.
        if self.parent_stack.valmode is True:
            blocks = [block for block in self.d_in if block['est'] is True]

        else:
            blocks = self.d_in

        # check if d_in has or not 'pred' to perform or skip calculations.
        if 'pred' in blocks[0].keys():
            self.has_pred = True
        else:
            self.has_pred = False

        # get the data, then slice the tones and asign to the right bin
        for ii, b in enumerate(blocks):

            stim = b['stim']  # input 3d array: 0d #streasm ; 1d #trials; 2d time
            stim = stim.astype(np.int16)  # input stim is in uint8 which is problematic for diff
            resp = b['resp']  # input 2d array: 0d #trials ; 1d time
            if self.has_pred:
                pred = b['pred']  # same shape as resp

            diff = np.diff(stim, axis=2)  # transform the binary stim into an edge array for easy onset/offset location

            # initializes dictionaries to store all tone typed for both actual and predicted response.
            resp_slice_dict = {'stream0Std': list(),
                               'stream0Dev': list(),
                               'stream0Ons': list(),
                               'stream1Std': list(),
                               'stream1Dev': list(),
                               'stream1Ons': list()}
            interval_dict = {'stream0Std': list(),
                             'stream0Dev': list(),
                             'stream0Ons': list(),
                             'stream1Std': list(),
                             'stream1Dev': list(),
                             'stream1Ons': list()}
            resp_spont = list()

            if self.has_pred:
                pred_slice_dict = {'stream0Std': list(),
                                   'stream0Dev': list(),
                                   'stream0Ons': list(),
                                   'stream1Std': list(),
                                   'stream1Dev': list(),
                                   'stream1Ons': list()}
                pred_spont = list()

            # define the length of the tones, assumes all tones are equal. defines flanking silences as with the same
            # length as the tone
            # this infers the tone length from the envelope shape, overlaping tones will give problems.
            adiff = diff[0, 0, :]
            IdxStrt = np.where(adiff == 1)[0][0]
            IdxEnd = np.where(adiff == -1)[0][0]
            toneLen = IdxEnd - IdxStrt

            for trialcounter in range(stim.shape[1]):

                # get starting indexes for both streams

                where0 = np.where(diff[0, trialcounter, :] == 1)[0] + 1
                where1 = np.where(diff[1, trialcounter, :] == 1)[0] + 1

                # calculates the time interval between consecutive tones of the same type
                time0 = where0.copy()
                time1 = where1.copy()
                time0[1:] = np.diff(time0)
                time1[1:] = np.diff(time1)

                # get the index of the onset tone, if one stream lacks tones then sets the index to max by default
                # so the other stream necessarily has the onset.
                try:
                    first0 = where0[0]
                except:
                    first0 = stim.shape[2]

                try:
                    first1 = where1[0]
                except:
                    first1 = stim.shape[2]

                # slices both streams

                # this cool list comprehension should work, if trials didnt terminate prematurely
                # respstream0 = [resp[trialcounter, ii - toneLen: ii + (toneLen * 2)] for ii in where0]
                # respstream1 = [resp[trialcounter, ii - toneLen: ii + (toneLen * 2)] for ii in where1]

                respstream0 = list()
                for ii in where0:
                    single_tone = np.empty(toneLen * 3)
                    single_tone[:] = np.NAN
                    tone = resp[trialcounter, ii - toneLen: ii + (toneLen * 2)]
                    single_tone[0:len(tone)] = tone
                    respstream0.append(single_tone)

                respstream1 = list()
                for ii in where1:
                    single_tone = np.empty(toneLen * 3)
                    single_tone[:] = np.NAN
                    tone = resp[trialcounter, ii - toneLen: ii + (toneLen * 2)]
                    single_tone[0:len(tone)] = tone
                    respstream1.append(single_tone)

                if self.has_pred:

                    # predstream0 = [pred[trialcounter, ii - toneLen: ii + (toneLen * 2)] for ii in where0]
                    # predstream1 = [pred[trialcounter, ii - toneLen: ii + (toneLen * 2)] for ii in where1]

                    predstream0 = list()
                    for ii in where0:
                        single_tone = np.empty(toneLen * 3)
                        single_tone[:] = np.NAN
                        tone = pred[trialcounter, ii - toneLen: ii + (toneLen * 2)]
                        single_tone[0:len(tone)] = tone
                        predstream0.append(single_tone)

                    predstream1 = list()
                    for ii in where1:
                        single_tone = np.empty(toneLen * 3)
                        single_tone[:] = np.NAN
                        tone = pred[trialcounter, ii - toneLen: ii + (toneLen * 2)]
                        single_tone[0:len(tone)] = tone
                        predstream1.append(single_tone)

                # checks which comes first and extract onset and preceding silence for spontaneous activity cuantification

                if first0 < first1:
                    # Onset is in stream 0
                    resp_slice_dict['stream0Ons'] = resp_slice_dict['stream0Ons'] + [respstream0[0]]
                    respstream0 = respstream0[1:]
                    interval_dict['stream0Ons'].append(time0[0])
                    time0 = time0[1:]
                    # actual silence to onset
                    resp_spont.append(resp[trialcounter, :where0[0]])

                    if self.has_pred:
                        pred_slice_dict['stream0Ons'] = pred_slice_dict['stream0Ons'] + [predstream0[0]]
                        predstream0 = predstream0[1:]
                        # predicted silence to onset
                        pred_spont.append(pred[trialcounter, :where0[0]])

                elif first0 > first1:
                    # Onset in in stream 1
                    resp_slice_dict['stream1Ons'] = resp_slice_dict['stream1Ons'] + [respstream1[0]]
                    respstream1 = respstream1[1:]
                    interval_dict['stream1Ons'].append(time1[0])
                    time1 = time1[1:]
                    # actual silence to onset
                    resp_spont.append(resp[trialcounter, :where1[0]])

                    if self.has_pred:
                        pred_slice_dict['stream1Ons'] = pred_slice_dict['stream1Ons'] + [predstream1[0]]
                        predstream1 = predstream1[1:]
                        # predicted silence to onset
                        pred_spont.append(pred[trialcounter, :where1[0]])

                # Count tones by integration
                tone_count = np.nansum(stim[:, trialcounter, :], axis=1)

                # Check which stream is standard and appends slices and intervals in the right list
                if tone_count[0] > tone_count[1]:
                    # stream 0 is standard, stream 1 is deviant
                    resp_slice_dict['stream0Std'] = resp_slice_dict['stream0Std'] + respstream0
                    resp_slice_dict['stream1Dev'] = resp_slice_dict['stream1Dev'] + respstream1
                    interval_dict['stream0Std'] += time0.tolist()
                    interval_dict['stream1Dev'] += time1.tolist()

                    if self.has_pred:
                        pred_slice_dict['stream0Std'] = pred_slice_dict['stream0Std'] + predstream0
                        pred_slice_dict['stream1Dev'] = pred_slice_dict['stream1Dev'] + predstream1

                elif tone_count[0] < tone_count[1]:
                    # Stream 1 is standard, stream 0 is deviant

                    resp_slice_dict['stream1Std'] = resp_slice_dict['stream1Std'] + respstream1
                    resp_slice_dict['stream0Dev'] = resp_slice_dict['stream0Dev'] + respstream0
                    interval_dict['stream1Std'] += time1.tolist()
                    interval_dict['stream0Dev'] += time0.tolist()

                    if self.has_pred:
                        pred_slice_dict['stream1Std'] = pred_slice_dict['stream1Std'] + predstream1
                        pred_slice_dict['stream0Dev'] = pred_slice_dict['stream0Dev'] + predstream0

            # calculates activity for each slice pool: first averages across trials, then integrates from the start
            # of the tone to the end of the slice. Organizes in an Activity dictionary with the same keys
            all_resp_tone_types = resp_slice_dict.copy()  # holds a copy por all activity calculation
            resp_slice_dict = {key: np.asarray(value) for key, value in resp_slice_dict.items()}
            resp_tone_act_dict = {key: np.nansum(value[:, toneLen:], axis=1)
                                  for key, value in resp_slice_dict.items()}
            resp_act_dict = {key: np.nansum(np.nanmean(value, axis=0)[toneLen:])
                             for key, value in resp_slice_dict.items()}

            # repeats the same as last for predicted responses if any
            if self.has_pred:
                all_pred_tone_types = pred_slice_dict.copy()  # holds a copy por all activity calculation
                pred_slice_dict = {key: np.asarray(value) for key, value in pred_slice_dict.items()}
                pred_tone_act_dict = {key: np.nansum(value[:, toneLen:], axis=1)
                                      for key, value in pred_slice_dict.items()}
                pred_act_dict = {key: np.nansum(np.nanmean(value, axis=0)[toneLen:])
                                 for key, value in pred_slice_dict.items()}

            # calculates the response ssa index  values for each stream and cell and organizes in a dictionary
            resp_SI_dict = {
                'stream0': (resp_act_dict['stream0Dev'] - resp_act_dict['stream0Std']) /  # dev - std over...
                           (resp_act_dict['stream0Dev'] + resp_act_dict['stream0Std']),  # dev + std

                'stream1': (resp_act_dict['stream1Dev'] - resp_act_dict['stream1Std']) /  # dev - std over...
                           (resp_act_dict['stream1Dev'] + resp_act_dict['stream1Std']),  # dev + std

                'cell': (resp_act_dict['stream0Dev'] + resp_act_dict['stream1Dev'] -  # dev + dev minus
                         resp_act_dict['stream0Std'] - resp_act_dict['stream1Std']) /  # std - std over
                        (resp_act_dict['stream0Dev'] + resp_act_dict['stream1Dev'] +  # dev + dev plus
                         resp_act_dict['stream0Std'] + resp_act_dict['stream1Std'])}  # std + std
            # does the same as last for predicted responses if any
            if self.has_pred:
                pred_SI_dict = {
                    'stream0': (pred_act_dict['stream0Dev'] - pred_act_dict['stream0Std']) /  # dev - std over...
                               (pred_act_dict['stream0Dev'] + pred_act_dict['stream0Std']),  # dev + std

                    'stream1': (pred_act_dict['stream1Dev'] - pred_act_dict['stream1Std']) /  # dev - std over...
                               (pred_act_dict['stream1Dev'] + pred_act_dict['stream1Std']),  # dev + std

                    'cell': (pred_act_dict['stream0Dev'] + pred_act_dict['stream1Dev'] -  # dev + dev minus
                             pred_act_dict['stream0Std'] - pred_act_dict['stream1Std']) /  # std - std over
                            (pred_act_dict['stream0Dev'] + pred_act_dict['stream1Dev'] +  # dev + dev plus
                             pred_act_dict['stream0Std'] + pred_act_dict['stream1Std'])}  # std + std

            # transformt the spontaneous activity list of heterogeneous lists into a 2d array padded with nan
            def aspadedarray(v, fillval=np.nan):
                lens = np.array([len(item) for item in v])
                mask = lens[:, None] > np.arange(lens.max())
                out = np.full(mask.shape, fillval)
                out[mask] = np.concatenate(v)
                return out

            resp_spont = aspadedarray(resp_spont, np.nan)
            if self.has_pred:
                pred_spont = aspadedarray(pred_spont, np.nan)

            # organizes the ssa index data into a dictionary containing the SI of the response and of the prediction in
            # corresponding keys, then append to the block list.
            # also append block dependent calculations into lists for such elements across all blocks of one cell.

            block_SI_dict = dict()

            block_SI_dict['resp_SI'] = resp_SI_dict
            folded_resp.append(resp_slice_dict)
            resp_tone_act.append(resp_tone_act_dict)
            interval_dict = {key: np.asarray(value) for key, value in interval_dict.items()}
            intervals.append(interval_dict)
            resp_SI.append(resp_SI_dict)
            cell_resp_spont.append(resp_spont)

            if self.has_pred:
                block_SI_dict['pred_SI'] = pred_SI_dict
                pred_tone_act.append(pred_tone_act_dict)
                folded_pred.append(pred_slice_dict)
                pred_SI.append(pred_SI_dict)
                cell_pred_spont.append(pred_spont)

            out_SI_dicts.append(block_SI_dict)

            # calculates the stream activitiy for all tones, and calculates the activity ratio between the streams
            # pools all the tones into the respective stream, regardless onset, standard or deviant.
            all_resp_act = [list(), list()]
            for key, value in all_resp_tone_types.items():
                if self.z_score == 'all':
                    if key[:7] == 'stream0':
                        all_resp_act[0] += value
                    elif key[:7] == 'stream1':
                        all_resp_act[1] += value

                elif self.z_score == 'spont':
                    if key == 'stream0Std':
                        all_resp_act[0] += value
                    elif key == 'stream1Std':
                        all_resp_act[1] += value

            # Then calculates the activity
            for ii, stream in enumerate(all_resp_act):
                stream = np.asarray(stream)
                if self.z_score == 'all':
                    all_resp_act[ii] = (np.nanmean(np.nanmean(stream, axis=0)[toneLen:])) * np.nanmean(
                        resp) / np.nanstd(resp)
                elif self.z_score == 'spont':
                    all_resp_act[ii] = ((np.nanmean(stream[:, toneLen:])) - np.nanmean(resp_spont)) / (
                                        np.nanstd(np.nanmean(stream[:, toneLen:], axis = 1)))

            # creates a dictionary and appends it to the block list
            all_resp_act_dict = {'stream0': all_resp_act[0],
                                 'stream1': all_resp_act[1],
                                 'ratio': np.min(all_resp_act) / np.max(all_resp_act),
                                 'mean': np.nanmean(all_resp_act)}

            # also calculates activity in the same way for the predicted responses.
            if self.has_pred:
                pred_spont_mean = np.nanmean(np.asarray(pred_spont))
                all_pred_act = [list(), list()]
                for key, value in all_pred_tone_types.items():
                    if self.z_score == 'all':
                        if key[:7] == 'stream0':
                            all_pred_act[0] += value
                        elif key[:7] == 'stream1':
                            all_pred_act[1] += value

                    elif self.z_score == 'spont':
                        if key == 'stream0Std':
                            all_pred_act[0] += value
                        elif key == 'stream1Std':
                            all_pred_act[1] += value

                # Then calculates the activity
                for ii, stream in enumerate(all_pred_act):
                    stream = np.asarray(stream)
                    if self.z_score == 'all':
                        all_pred_act[ii] = np.nanmean(np.nanmean(np.asarray(stream), axis=0)[toneLen:]) * np.nanmean(
                            pred) / np.nanstd(pred)
                    elif self.z_score == 'spont':
                        all_pred_act[ii] = ((np.nanmean(stream[:, toneLen:])) - np.nanmean(pred_spont)) / (
                                            np.nanstd(np.nanmean(stream[:, toneLen:], axis = 1)))

                # creates a dictionary and appends it to the block list
                all_pred_act_dict = {'stream0': all_pred_act[0],
                                     'stream1': all_pred_act[1],
                                     'ratio': np.min(all_pred_act) / np.max(all_pred_act),
                                     'mean': np.nanmean(all_pred_act)}


            block_act_dict = dict()
            block_act_dict['resp_act'] = all_resp_act_dict
            if self.has_pred:
                block_act_dict['pred_act'] = all_pred_act_dict

            out_act_dicts.append(block_act_dict)

        self.folded_resp = folded_resp
        self.resp_tone_act = resp_tone_act
        self.intervals = intervals
        self.resp_SI = resp_SI
        self.parent_stack.meta['ssa_index'] = out_SI_dicts
        self.stream_activity = out_act_dicts
        self.resp_spont = cell_resp_spont

        if self.has_pred:
            self.folded_pred = folded_pred
            self.pred_tone_act = pred_tone_act
            self.pred_SI = pred_SI
            self.pred_spont = cell_pred_spont

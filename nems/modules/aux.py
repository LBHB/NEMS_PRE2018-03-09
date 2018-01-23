#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auxiliary modules for fitting.

Created on Fri Aug  4 14:00:03 2017

@author: shofer
"""

import logging
log = logging.getLogger(__name__)

import numpy as np
import copy

from nems.modules.base import nems_module
import nems.utilities.utils
import nems.utilities.plot

class psth(nems_module):
    """ compute PSTH for each unique stimulus and substiute as "pred" for each
    single-trial response.  Useful for baseline/gain analysis without an
    explicit STRF"""
    
    name = 'aux.psth'
    plot_fns = [nems.utilities.plot.sorted_raster,
                nems.utilities.plot.raster_plot,
                nems.utilities.plot.plot_stim_psth]
    """
    Replaces stim with average resp for each stim. This is the 'perfect' model
    used for comparing different models of pupil state gain.
    """

    def my_init(self):
        log.info('aux.psth: Replacing pred with PSTH response')

    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
            
        output_name=self.output_name
        psth={}
        for f_in, f_out in zip(self.d_in, self.d_out):
            stimset=np.unique(np.array(f_in['replist']))
            f_out[output_name]=f_in['resp'].copy()
            
            for stimidx in stimset:
                i = np.array(f_in['replist'])[:,0]==stimidx
                if f_in['est']:
                    # compute PSTH for estimation data
                    psth[stimidx]=np.mean(f_in['resp'][:,i,:],axis=1,keepdims=True)
                # set predcition to est PSTH for both est and val data
                f_out[output_name][:,i,:]=psth[stimidx]
            

class normalize(nems_module):
    """
    normalize - rescale a variable, typically stim, to put it in a range that
    works well with fit algorithms --
    either mean 0, variance 1 (if sign doesn't matter) or
    min 0, max 1 (if positive values desired)

    IMPORTANT NOTE: normalization factors are computed from estimation data
    only but applied to both estimation and validation data streams
    """
    # TODO: it might be better to build this more intrinsically into the stack
    # object, or else it has to be appended in every keyword?

    # TODO: this is having issues with batch294 data used with perfectpupil50?
    # Not sure why, it works fine for nested and non-nested crossval otherwise
    #---this definitely has something to do with where this module is appended
    # in the stack.

    name = 'aux.normalize'
    user_editable_fields = ['input_name', 'output_name', 
                            'force_positive','norm_gain', 'norm_base']
    force_positive = True
    norm_gain=1
    norm_base=0
    
    def my_init(self, force_positive=True, num_channels=None):
        self.auto_plot = False
        if num_channels is None:
            num_channels = self.d_in[0][self.input_name].shape[0]
        self.norm_gain=np.ones([num_channels,1])
        self.norm_base=np.zeros([num_channels,1])
        self.force_positive = force_positive

    def evaluate(self):
        del self.d_out[:]
        # create a copy of each input variable
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
            
        Z=self.unpack_data(self.input_name,est=True)
        
        # compute std() of est data output and then normalize
        if self.force_positive:
            self.norm_base=np.min(Z,axis=1,keepdims=True)
        else:
            self.norm_base=np.mean(Z,axis=1,keepdims=True)    
        self.norm_gain=np.std(Z,axis=1,keepdims=True)
        
        Z=(Z-self.norm_base)/self.norm_gain
        self.pack_data(Z,self.output_name,est=True)
        
        if self.parent_stack.valmode:
            Z=self.unpack_data(self.input_name,est=False)
            # don't recalc. just use baseline/gain from est data
            Z=(Z-self.norm_base)/self.norm_gain
            self.pack_data(Z,self.output_name,est=False)
            
    def evaluate_old(self, nest=0):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            # create a copy of each input variable
            self.d_out.append(copy.copy(d))

        X = self.unpack_data(name=self.input_name, est=True, use_dout=False)
        if self.force_positive:
            self.d = X.min(axis=-1)
            m = (X.max(axis=-1).T - self.d.T).T
            m[m == 0] = 1
            self.g = 1 / m
        else:
            self.d = X.mean(axis=-1)
            self.g = X.std(axis=-1)

        for f_in, f_out in zip(self.d_in, self.d_out):
            # don't need to eval the est data for each nest, just the first one
            X = copy.deepcopy(f_in[self.input_name])
            f_out[self.output_name] = ((X.T - self.d) * self.g).T

        if hasattr(self, 'state_mask'):
            del_idx = []
            for i in range(0, len(self.d_out)):
                if not self.d_out[i]['filestate'] in self.state_mask:
                    del_idx.append(i)
            for i in sorted(del_idx, reverse=True):
                del self.d_out[i]


class add_scalar(nems_module):
    """
    add_scalar -- pretty much a dummy test module but may be useful for
    some reason
    """
    name = 'aux.add_scalar'
    user_editable_fields = ['input_name', 'output_name', 'n']
    n = np.zeros([1, 1])

    def my_init(self, n=0, fit_fields=['n']):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.fit_fields = fit_fields
        self.n[0, 0] = n

    def my_eval(self, X):
        Y = X + self.n
        return Y


class dc_gain(nems_module):
    """
    dc_gain -- apply a scale and offset term
    """

    name = 'aux.dc_gain'
    user_editable_fields = ['input_name', 'output_name', 'd', 'g']
    d = np.zeros([1, 1])
    g = np.ones([1, 1])

    def my_init(self, d=0, g=1, fit_fields=['d', 'g']):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.fit_fields = fit_fields
        self.d[0, 0] = d
        self.g[0, 0] = g

    def my_eval(self, X):
        Y = X * self.g + self.d
        return Y


class sum_dim(nems_module):
    """
    sum_dim - sum a matrix across one dimension. maybe useful? mostly testing
    """
    name = 'aux.sum_dim'
    user_editable_fields = ['input_name', 'output_name', 'dim']
    dim = 0

    def my_init(self, dim=0):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.dim = dim
        # self.save_dict={'dim':dim}

    def my_eval(self, X):
        Y = X.sum(axis=self.dim)
        return Y


class onset_edges(nems_module):
    """
    onset_edges - calculate diff, replace positive diffs with 1, everything else with zero
    """
    name = 'aux.onset_edges'
    user_editable_fields = ['input_name', 'output_name', 'dim', 'state_mask']
    dim = 0
    state_mask = [0, 1]
    plot_fns = [nems.utilities.plot.plot_stim,
                nems.utilities.plot.plot_spectrogram]

    def my_init(self, dim=2, state_mask=[0, 1]):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.dim = dim
        self.state_mask = state_mask

    def my_eval(self, X):
        dim = self.dim
        s = list(X.shape)
        s[dim] = 1
        Z = np.zeros(s)
        Y = np.concatenate((Z, np.diff(X.astype(float))), axis=dim)
        Y[Y < 0] = 0

        return Y

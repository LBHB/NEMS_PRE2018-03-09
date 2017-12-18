#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Base nems_module class

Created on Fri Aug  4 12:49:40 2017

@author: shofer
"""


import numpy as np
import copy
#import nems.utilities.utils
import nems.utilities.plot
#import nems


# TODO: this should really be set up as a proper abstract base class at some point
# ---njs August 4 2017


class nems_module:
    """nems_module

    Generic NEMS module

    """

    #
    # common attributes for all modules
    #
    name = 'base.nems_module'
    user_editable_fields = ['input_name', 'output_name', 'fit_fields']
    plot_fns = [nems.utilities.plot.plot_spectrogram]

    input_name = 'pred'  # name of input matrix in d_in
    output_name = 'pred'  # name of output matrix in d_out
    state_var = 'pupil'
    parent_stack = None  # pointer to stack instance that owns this module
    idm = None  # unique name for this module to be referenced from the stack??
    # pointer to input of data stack, ie, for modules[i], parent_stack.d[i]
    d_in = None
    d_out = None  # pointer to output, parent_stack.d[i+!]
    fit_fields = []  # what fields should be fed to phi for fitting
    auto_plot = True  # whether to include in quick_plot
    save_dict = {}
    #
    # Begin standard functions
    #

    def __init__(self, parent_stack=None, input_name=None,
                 output_name=None, state_var=None, **xargs):
        """
        Standard initialization for all modules. Sets up next step in data
        stream linking parent_stack.data to self.d_in and self.d_out.
        Also configures default plotter and calls self.my_init(), which can
        optionally be defined to perform module-specific initialization.
        """
        print("creating module " + self.name)
        if parent_stack is None:
            self.d_in = []
        else:
            # point to parent in order to allow access to it attributes
            self.parent_stack = parent_stack
            # d_in is by default the last entry of parent_stack.data
            self.d_in = parent_stack.data[-1]
            self.idm = "{0}{1}".format(self.name, len(parent_stack.modules))
        if not input_name is None:
            self.input_name = input_name
        if not output_name is None:
            self.output_name = output_name
        if not state_var is None:
            self.state_var = state_var
        self.d_out = copy.deepcopy(self.d_in)
        self.auto_plot = True
        self.do_plot = self.plot_fns[0]  # default is first in list
        self.do_trial_plot = self.plot_fns[0]
        self.my_init(**xargs)
        # not sure that this is a complete list
        # self.user_editable_fields=['input_name','output_name']+list(self.field_dict.keys())

    def get_phi(self):
        return {k: getattr(self, k) for k in self.fit_fields}

    def set_phi(self, phi):
        for k, v in phi.items():
            setattr(self, k, v)

    def get_user_fields(self):
        f = {}
        print(self.user_editable_fields)
        for k in self.user_editable_fields:
            t = getattr(self, k)
            if isinstance(t, np.ndarray):
                t = t.tolist()
            f[k] = t
        return f

    def unpack_data(self, name='stim', est=True, use_dout=False):
        """
        unpack_data - extract a data variable from all files into a single
        matrix (concatenated across files)
        """
        m = self
        if use_dout:
            D = m.d_out
        else:
            D = m.d_in

        if D[0][name].ndim == 2:
            X = np.empty([1, 0])
            s = m.d_in[0][name].shape
        else:
            s = D[0][name].shape
            X = np.empty([s[0], 0])

        for i, d in enumerate(D):
            if not 'est' in d.keys():
                if d[name].ndim == 2:
                    X = np.concatenate(
                        (X, d[name].reshape([1, -1], order='C')), axis=1)
                else:
                    X = np.concatenate(
                        (X, d[name].reshape([s[0], -1], order='C')), axis=1)
            elif (est and d['est']):
                if d[name].ndim == 2:
                    X = np.concatenate(
                        (X, d[name].reshape([1, -1], order='C')), axis=1)
                else:
                    X = np.concatenate(
                        (X, d[name].reshape([s[0], -1], order='C')), axis=1)
            elif not est and not d['est']:
                if d[name].ndim == 2:
                    X = np.concatenate(
                        (X, d[name].reshape([1, -1], order='C')), axis=1)
                else:
                    X = np.concatenate(
                        (X, d[name].reshape([s[0], -1], order='C')), axis=1)

        return X

    def pack_data(self, X, name='stim', est=True, use_dout=True):
        """
        unpack_data - extract a data variable from all files into a single
        matrix (concatenated across files)
        """
        m = self
        if use_dout:
            D = m.d_out
        else:
            D = m.d_in

        s = X.shape
        for i, d in enumerate(D):
            if not 'est' in d.keys() or (est and d['est']) or (
                    not est and not d['est']):
                s2 = list(d[name].shape)
                s2[0] = s[0]
                n = np.prod(s2[1:])
                d[name] = np.reshape(X[:, 0:n], s2)
                X = X[:, n:]

    def evaluate(self, nest=0):
        """
        General evaluate function, for both nested and non-nested crossval. Creates
        a copy of the d_in dataframe and places it in the next position in stack.data.
        Then calls the module specific my_eval, and replaces d_out[output_name] with
        the output of my_eval.
        """
        if nest == 0:
            del self.d_out[:]
            # create a copy of each input variable
            for i, d in enumerate(self.d_in):
                # self.d_out.append(copy.deepcopy(d))
                # TODO- make it so don't deepcopy eveything. deal with nesting!
                self.d_out.append(copy.copy(d))

        for f_in, f_out in zip(self.d_in, self.d_out):
            if self.parent_stack.nests > 0 and f_in['est'] is False:
                X = copy.deepcopy(f_in[self.input_name][nest])
                # duplicate placeholder list in case output_name is a new
                # variable
                if nest == 0:
                    print("nest={0} deep copying in[{1}] to out[{2}]".format(
                        nest, self.input_name, self.output_name))
                    f_out[self.output_name] = copy.deepcopy(
                        f_in[self.input_name])
                f_out[self.output_name][nest] = self.my_eval(X)
            elif nest == 0:
                # don't need to eval the est data for each nest, just the first
                # one
                X = copy.deepcopy(f_in[self.input_name])
                f_out[self.output_name] = self.my_eval(X)

        if hasattr(self, 'state_mask'):
            del_idx = []
            for i in range(0, len(self.d_out)):
                if not self.d_out[i]['filestate'] in self.state_mask:
                    del_idx.append(i)
            for i in sorted(del_idx, reverse=True):
                del self.d_out[i]

    def my_init(self, **xargs):
        """
        Placeholder for module specific initialization. my_init is defined for each
        module (with some specific exceptions).
        """
        pass

    def my_eval(self, X):
        """
        Placeholder for module-specific evaluation, default is
        pass-through of pointer to input data matrix.
        """
        Y = X
        return Y

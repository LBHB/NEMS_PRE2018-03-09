#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules for integrating state (task, performance, pupil) in encoding models

Created on Fri Aug  4 13:29:30 2017

"""

import logging
log = logging.getLogger(__name__)

from nems.modules.base import nems_module
import nems.utilities.utils
import nems.utilities.plot

import numpy as np
import copy
import scipy.special as sx


class state_gain(nems_module):
    """
    state_gain - apply a gain/offset based on continuous pupil diameter, or some
    other continuous variable. Does not use standard my_eval, instead uses its own
    evaluate() that overrides the nems_module evaluate()
    """
    
    name = 'state.state_gain'
    user_editable_fields = ['input_name', 'output_name',
                            'fit_fields', 'state_var', 'gain_type', 'theta']
    gain_type = 'lingain'
    plot_fns = [nems.utilities.plot.state_act_scatter_smooth, nems.utilities.plot.pre_post_psth,
                nems.utilities.plot.pred_act_psth_all, nems.utilities.plot.non_plot]

    def my_init(self, gain_type='lingain', fit_fields=['theta'], theta=[0, 1, 0, 0],
                order=None):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.fit_fields = fit_fields
        self.gain_type = gain_type
        self.theta = np.array([theta])
        self.order = order
        self.do_plot = self.plot_fns[0]

    def nogain_fn(self, X, Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve
        state variable. This is the "control" for the state_gain exploration.
        """
        Y = self.theta[0, 0] + self.theta[0, 1] * X
        return(Y)

    def lingainctl_fn(self, X, Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve
        state variable. This is the "control" for the state_gain exploration.

        SVD mod: shuffle state, keep same number of parameters for proper control
        """

        # only apply to dims of Xp spanned by theta
        dims=np.int(self.theta.shape[1]/2-1)
        Xp=Xp[0:dims,:]
        
        # OLD WAY -- not random enough
        if 0:
            s = Xp.shape
            n = np.int(np.ceil(s[0] / 2))
            # log.info(s)
            # log.info(n)
            # log.info(Xp.shape)
            Xp = np.roll(Xp, n, 0)
        else:
            # save current random state
            prng = np.random.RandomState()
            save_state = prng.get_state()
            prng = np.random.RandomState(1234567890)

            # shuffle state vector across trials (time)
            for ii in range(0, Xp.shape[0]):
                fxp = np.isfinite(Xp[ii, :])
                txp = Xp[ii, fxp]
                prng.shuffle(txp)
                Xp[ii, fxp] = txp

            # prng.shuffle(Xp)

            # restore saved random state
            prng.set_state(save_state)

        Y, Xp = self.lingain_fn(X, Xp)
        return Y, Xp

    def lingain_fn(self, X, Xp):
        # expect theta = [b0 g0 b1 p1 b2 p2...] where 1, 2 are first dimension
        # of Xp (pupil, behavior state, etc)
        
        # only apply to dims of Xp spanned by theta
        dims=np.int(self.theta.shape[1]/2-1)
        Xp=Xp[0:dims,:]

        Y = self.theta[0, 0] + (self.theta[0, 1] * X)
        for ii in range(0, Xp.shape[0]):
            Y += (self.theta[0, 2 + ii * 2] * Xp[ii, :]) + \
                self.theta[0, 3 + ii * 2] * np.multiply(Xp[ii, :], X)
        return Y, Xp

    def expgain_fn(self, X, Xp):
        Y = self.theta[0, 0] + self.theta[0, 1] * X * \
            np.exp(self.theta[0, 2] * Xp + self.theta[0, 3])
        return(Y)

    def loggain_fn(self, X, Xp):
        Y = self.theta[0, 0] + self.theta[0, 1] * X * \
            np.log(self.theta[0, 2] + Xp + self.theta[0, 3])
        return(Y)

    def polygain_fn(self, X, Xp):
        """
        Fits a polynomial gain function:
        Y = g0 + g*X + d1*X*Xp^1 + d2*X*Xp^2 + ... + d(n-1)*X*Xp^(n-1) + dn*X*Xp^n
        """
        deg = self.theta.shape[1]
        Y = 0
        for i in range(0, deg - 2):
            Y += self.theta[0, i] * X * np.power(Xp, i + 1)
        Y += self.theta[0, -2] + self.theta[0, -1] * X
        return(Y)

    def powergain_fn(self, X, Xp):
        """
        Slightly different than polypugain. Y = g0 + g*X + d0*Xp^n + d*X*Xp^n
        """
        deg = self.order
        v = self.theta
        Y = v[0, 0] + v[0, 1] * X + v[0, 2] * \
            np.power(Xp, deg) + v[0, 3] * np.multiply(X, np.power(Xp, deg))
        return(Y)

    def Poissongain_fn(self, X, Xp):  # Kinda useless, might delete ---njs
        u = self.theta[0, 1]
        Y = self.theta[0, 0] * X * \
            np.divide(np.exp(-u) * np.power(u, Xp), sx.factorial(Xp))
        return(Y)

    def butterworthHP_fn(self, X, Xp):
        """
        Applies a Butterworth high pass filter to the state data, with a DC offset.
        state diameter is treated here as analogous to frequency, and the fitted
        parameters are DC offset, overall gain, and f3dB. Order is specified, and
        controls how fast the rolloff is.
        """
        n = self.order
        Y = self.theta[0, 2] + self.theta[0, 0] * X * np.divide(np.power(np.divide(Xp, self.theta[0, 1]), n),
                                                                np.sqrt(1 + np.power(np.divide(Xp, self.theta[0, 1]), 2 * n)))
        return(Y)

    def evaluate(self):
        m = self
        del m.d_out[:]
        for i, d in enumerate(m.d_in):
            # self.d_out.append(copy.deepcopy(val))
            m.d_out.append(d.copy())

        X = m.unpack_data(name=m.input_name, est=True)
        Xp = m.unpack_data(name=m.state_var, est=True)
        Z, Zp = getattr(m, m.gain_type + '_fn')(X, Xp)
        m.pack_data(Z, name=m.input_name, est=True)
        m.pack_data(Zp, name=m.state_var, est=True)

        if m.parent_stack.valmode:
            X = m.unpack_data(name=m.input_name, est=False)
            Xp = m.unpack_data(name=m.state_var, est=False)
            Z, Zp = getattr(m, m.gain_type + '_fn')(X, Xp)
            m.pack_data(Z, name=m.input_name, est=False)
            m.pack_data(Zp, name=m.state_var, est=False)


class state_weight(nems_module):
    """
    combined weighting of two predicted PSTHs, depending on state_var
    
    @author: svd
    """

    name = 'state.state_weight'
    user_editable_fields = ['input_name', 'input_name2', 'output_name',
                            'fit_fields', 'state_var', 'input_name2', 'weight_type', 'theta']
    weight_type = 'linear'
    plot_fns = [nems.utilities.plot.state_act_scatter_smooth, nems.utilities.plot.pre_post_psth,
                nems.utilities.plot.pred_act_psth_all, nems.utilities.plot.non_plot]
    input_name2 = 'pred2'
    theta = np.zeros([1, 2])

    def my_init(self, input_name2="pred2",
                weight_type='linear', fit_fields=['theta'], theta=[.1, .1]):
        self.input_name2 = input_name2
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.fit_fields = fit_fields
        self.weight_type = weight_type
        self.my_eval = getattr(self, self.weight_type + '_fn')
        self.theta = np.array([theta])
        self.do_plot = self.plot_fns[0]

    def linear_fn(self, X1, X2, Xp):
        """
        linear weighting of two predicted PSTHs, depending on state_var
        w= a + b * p(t)  hard bounded at 0 and 1
        """
        w = self.theta[0, 0] + self.theta[0, 1] * Xp[0,:]
        w[w < 0] = 0
        w[w > 1] = 1
        Y = (1 - w) * X1 + w * X2
        return(Y, Xp)

    def linearctl_fn(self, X1, X2, Xp):
        """
        shuffle state, keep same number of parameters for proper control
        """

        # save current random state
        prng = np.random.RandomState()
        save_state = prng.get_state()
        prng = np.random.RandomState(1234567890)

        # shuffle state vector across trials (time)
        for ii in range(0, Xp.shape[0]):
            fxp = np.isfinite(Xp[ii, :])
            txp = Xp[ii, fxp]
            prng.shuffle(txp)
            Xp[ii, fxp] = txp

        # restore saved random state
        prng.set_state(save_state)

        Y, Xp = self.linear_fn(X1, X2, Xp)

        return(Y, Xp)

    def evaluate(self, nest=0):
        m = self
        del m.d_out[:]
        for i, d in enumerate(m.d_in):
            # self.d_out.append(copy.deepcopy(val))
            m.d_out.append(copy.copy(d))

        X1 = m.unpack_data(name=m.input_name, est=True)
        X2 = m.unpack_data(name=m.input_name2, est=True)
        Xp = m.unpack_data(name=m.state_var, est=True)
        Y, Xp = self.my_eval(X1, X2, Xp)
        m.pack_data(Y, name=m.output_name, est=True)
        m.pack_data(Xp, name=m.state_var, est=True)

        if m.parent_stack.valmode:
            X1 = m.unpack_data(name=m.input_name, est=False)
            X2 = m.unpack_data(name=m.input_name2, est=False)
            Xp = m.unpack_data(name=m.state_var, est=False)
            Y, Xp = self.my_eval(X1, X2, Xp)
            m.pack_data(Y, name=m.output_name, est=False)
            m.pack_data(Xp, name=m.state_var, est=False)


class state_filter(nems_module):
    """
    apply some sort of transformation to state variable
    @author: svd
    """

    name = 'state.state_filter'
    user_editable_fields = ['input_name', 'output_name', 'fit_fields',
                            'state_var', 'input_name2', 'weight_type', 'theta']
    filter_type = 'slope'
    plot_fns = [nems.utilities.plot.pre_post_psth, nems.utilities.plot.state_act_scatter_smooth,
                nems.utilities.plot.pred_act_psth_all, nems.utilities.plot.non_plot]
    theta = np.zeros([1, 2])

    def my_init(self, filter_type='linear'):
        self.filter_type = filter_type
        self.my_eval = getattr(self, self.filter_type + '_fn')
        self.do_plot = self.plot_fns[0]

    def slope_fn(self, Xp):
        """
        linear weighting of two predicted PSTHs, depending on state_var
        w= a + b * p(t)  hard bounded at 0 and 1
        """

        slope = (np.mean(Xp[:, -30:-10], axis=1) -
                 np.mean(Xp[:, 10:30], axis=1))
        slope = np.reshape(slope, [-1, 1])
        Y = np.repeat(slope, Xp.shape[1], axis=1)
        return(Y)

    def slopectl_fn(self, X1, X2, Xp):
        """
        shuffle state, keep same number of parameters for proper control
        """

        # save current random state
        prng = np.random.RandomState()
        save_state = prng.get_state()
        prng = np.random.RandomState(1234567890)

        # shuffle state vector across trials (time)
        prng.shuffle(Xp)

        # restore saved random state
        prng.set_state(save_state)

        # s=Xp.shape
        # n=np.int(np.ceil(s[0]/2))
        # Xp=np.roll(Xp,n,0)

        Y = self.slope_fn(Xp)

        return Y

    def evaluate(self, nest=0):
        if nest == 0:
            del self.d_out[:]
            for i, val in enumerate(self.d_in):
                self.d_out.append(copy.deepcopy(val))
        for f_in, f_out in zip(self.d_in, self.d_out):
            if f_in['est'] is False:
                Xp = copy.deepcopy(f_in[self.input_name][nest])
                Xp = self.my_eval(Xp)
                f_out[self.output_name][nest] = Xp
            else:
                Xp = copy.deepcopy(f_in[self.input_name])
                Xp = self.my_eval(Xp)
                f_out[self.output_name] = Xp
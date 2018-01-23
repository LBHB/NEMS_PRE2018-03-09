#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modules for manipulating pupil data

ALL DEPRECATED AND CAN BE DELETED???

Created on Fri Aug  4 13:29:30 2017

@author: shofer
"""

import logging
log = logging.getLogger(__name__)

from nems.modules.base import nems_module
import nems.utilities.utils
import nems.utilities.plot

import numpy as np
import scipy.special as sx


class model(nems_module):
    name = 'pupil.model'
    plot_fns = [nems.utilities.plot.sorted_raster,
                nems.utilities.plot.raster_plot,
                nems.utilities.plot.plot_stim_psth]
    """
    DEPRECATED. REPLACED by aux.psth. DELETE ME?!?
    
    Replaces stim with average resp for each stim. This is the 'perfect' model
    used for comparing different models of pupil state gain.
    """

    def my_init(self):
        log.info('Replacing pred with averaged response raster')

    def evaluate(self, nest=0):
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
            
            # deprecated code from Shofer
            #Xa = f_in['avgresp']
            #R = f_in['replist']
            #X = np.squeeze(Xa[R, :])
            ## X=np.zeros(f_in['resp'].shape)
            ## for i in range(0,R.shape[0]):
            ##    X[i,:]=Xa[R[i],:]
            #f_out[self.output_name] = X[np.newaxis,:,:]


class pupgain(nems_module):
    """
    state_gain - apply a gain/offset based on continuous pupil diameter, or some
    other continuous variable. Does not use standard my_eval, instead uses its own
    evaluate() that overrides the nems_module evaluate()

    DEPRECATED? REPLACED BY state.state_gain
    """
    # Changed to helper function based general module --njs June 29 2017
    name = 'pupil.pupgain'
    user_editable_fields = ['input_name', 'output_name',
                            'fit_fields', 'state_var', 'gain_type', 'theta']
    gain_type = 'linpupgain'
    plot_fns = [nems.utilities.plot.state_act_scatter_smooth, nems.utilities.plot.pre_post_psth,
                nems.utilities.plot.pred_act_psth_all, nems.utilities.plot.non_plot]

    def my_init(self, gain_type='linpupgain', fit_fields=['theta'], theta=[0, 1, 0, 0],
                order=None):
        self.field_dict = locals()
        self.field_dict.pop('self', None)
        self.fit_fields = fit_fields
        self.gain_type = gain_type
        self.theta = np.array([theta])
        self.order = order
        self.do_plot = self.plot_fns[0]

    def nopupgain_fn(self, X, Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve
        state variable. This is the "control" for the state_gain exploration.
        """
        Y = self.theta[0, 0] + self.theta[0, 1] * X
        return(Y)

    def linpupgainctl_fn(self, X, Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve
        state variable. This is the "control" for the state_gain exploration.

        SVD mod: shuffle pupil, keep same number of parameters for proper control
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

        Y, Xp = self.linpupgain_fn(X, Xp)
        return Y, Xp

    def linpupgain_fn(self, X, Xp):
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

    def exppupgain_fn(self, X, Xp):
        Y = self.theta[0, 0] + self.theta[0, 1] * X * \
            np.exp(self.theta[0, 2] * Xp + self.theta[0, 3])
        return(Y)

    def logpupgain_fn(self, X, Xp):
        Y = self.theta[0, 0] + self.theta[0, 1] * X * \
            np.log(self.theta[0, 2] + Xp + self.theta[0, 3])
        return(Y)

    def polypupgain_fn(self, X, Xp):
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

    def powerpupgain_fn(self, X, Xp):
        """
        Slightly different than polypugain. Y = g0 + g*X + d0*Xp^n + d*X*Xp^n
        """
        deg = self.order
        v = self.theta
        Y = v[0, 0] + v[0, 1] * X + v[0, 2] * \
            np.power(Xp, deg) + v[0, 3] * np.multiply(X, np.power(Xp, deg))
        return(Y)

    def Poissonpupgain_fn(self, X, Xp):  # Kinda useless, might delete ---njs
        u = self.theta[0, 1]
        Y = self.theta[0, 0] * X * \
            np.divide(np.exp(-u) * np.power(u, Xp), sx.factorial(Xp))
        return(Y)

    def butterworthHP_fn(self, X, Xp):
        """
        Applies a Butterworth high pass filter to the pupil data, with a DC offset.
        Pupil diameter is treated here as analogous to frequency, and the fitted
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



import logging
log = logging.getLogger(__name__)

import numpy as np
import copy
import nems.utilities.plot


class nems_module:
    """nems_module

    Generic NEMS module

    """
    name = 'base.nems_module'
    user_editable_fields = ['input_name', 'output_name', 'fit_fields']
    plot_fns = [nems.utilities.plot.plot_spectrogram]
    fit_fields = []  # what fields should be fed to phi for fitting
    auto_plot = True  # whether to include in quick_plot
    save_dict = {}
    norm_output = False

    def __init__(self, parent_stack, **kwargs):
        log.info("creating module " + self.name)

        # point to parent in order to allow access to it attributes
        self.parent_stack = parent_stack
        # d_in is by default the last entry of parent_stack.data
        self.idm = "{0}{1}".format(self.name, len(parent_stack.modules))

        self.do_plot = self.plot_fns[0]
        self.do_trial_plot = self.plot_fns[0]

        # TODO: This is a hack to maintain certain aspects of NEMS that need to
        # go away eventually. For now, keep NEMS working ...
        for k, v in kwargs.items():
            setattr(self, k, v)

    def get_phi(self):
        return {k: getattr(self, k) for k in self.fit_fields}

    def set_phi(self, phi):
        for k, v in phi.items():
            setattr(self, k, v)

    def get_user_fields(self):
        f = {}
        log.info(self.user_editable_fields)
        for k in self.user_editable_fields:
            t = getattr(self, k)
            if isinstance(t, np.ndarray):
                t = t.tolist()
            f[k] = t
        return f

    def normalize(self, x, mode):
        # This allows us to recalculate the normalization factor when working
        # with estimation data, but not validation data. I suspect there's a
        # better way to do this, so let's think about it.
        if mode == 'est':
            self.norm_factor = np.std(np.abs(x), axis=1, keepdims=True)
        return x / self.norm_factor

    def init(self, recording):
        x = recording.get_signal(self.input_name)
        self.input_channels = x.shape[0]
        self.norm_factor = np.ones((self.input_channels, 1))

    def evaluate(self, recording, mode):
        '''
        Evaluate recording and return new recording with transformed signals

        Parameters
        ----------
        recording : instance of `Recording`
            Input recording to evaluate
        mode : {'est', 'val'}
            Indicates whether the recording contains estimation or validation
            data. This is used to determine whether we need to recompute certain
            factors such as the normalization value.

        Returns
        -------
        new_recording : instance of `Recording`
            This is a copy of the input recording containing the transformed
            signals (along with any untransformed signals from the original
            recording).
        '''

        raise NotImplementedError

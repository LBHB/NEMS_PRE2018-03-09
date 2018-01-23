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

    def __init__(self, parent_stack=None, input_name='pred', output_name='pred',
                 state_var='pupil', norm_output=False, **kwargs):

        log.info("creating module " + self.name)
        if parent_stack is not None:
            # point to parent in order to allow access to it attributes
            self.parent_stack = parent_stack
            # d_in is by default the last entry of parent_stack.data
            self.idm = "{0}{1}".format(self.name, len(parent_stack.modules))

        self.input_name = input_name
        self.output_name = output_name
        self.state_var = state_var
        self.auto_plot = True
        self.do_plot = self.plot_fns[0]  # default is first in list
        self.do_trial_plot = self.plot_fns[0]
        self.norm_output=norm_output

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
        if mode == 'est':
            self.norm_factor = np.std(np.abs(x), axis=1, keepdims=True)
        return x / self.norm_factor

    def init(self, recording):
        x = recording.get_signal(self.input_name)
        self.input_channels = x.shape[0]
        self.norm_factor = np.ones((self.input_channels, 1))

    def evaluate(self, recording_in, mode='est'):
        '''
        Defines a general evaluation function for modules with a single input
        and output. Override this method if you need to apply it to multiple
        inputs/outputs.
        '''
        # Pull out the named input, evaluate it and save the named output.
        x_signal = recording_in.get_signal(self.input_name)
        x = x_signal.as_continuous()
        y = self.simple_eval(x)
        if self.norm_output:
            y = self.normalize(x, mode)
        recording_out = recording_in.copy()
        y_signal = x_signal._modified_copy(y)
        recording_out.set_signal(self.output_name, y_signal)
        return recording_out

    def simple_eval(self, x):
        raise NotImplementedError

    def my_eval(self, X):
        """
        Placeholder for module-specific evaluation, default is
        pass-through of pointer to input data matrix.
        """
        raise NotImplementedError

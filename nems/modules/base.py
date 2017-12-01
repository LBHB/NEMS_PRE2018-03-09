import numpy as np
import copy


class nems_module:
    """ The NEMS module base class. Intended to be subclassed rather than
    used directly. In the __init__ method of your subclasses, it is
    required that you call the initialization of this method, like:

    def __init__(self, ...):
        super(MySubclassedModule, self).__init__()
        # Then your code here

    TODO: Field documentation
    """
    def __init__(self, parent_model,
                 name='base.nems_module',
                 input_signal_names=['stim'],
                 output_signal_name='pred',
                 state_var=None):
        self.name = name
        self.parent_model = parent_model  # The Model that owns this module
        self.input_signal_names = input_signal_names
        self.output_signal_name = output_signal_name
        self.d_in = self.parent_model.modules[-1].dout
        self.d_out = copy.copy(self.d_in)  # TODO: Shallow ok iff d_in is flat!
        self.fit_fields = []
        self.plot_fns = []
        self.auto_plot_idx = 0
        self.user_editable_fields = ['input_signal_names',
                                     'output_signal_name',
                                     'fit_fields']

    def parms2phi(self):
        """ Return a ndarray of all parameters that are in fields listed in
        self.fit_fields. Useful for passing params to fit routines. """
        phi = np.empty(shape=[0, 1])
        for k in self.fit_fields:
            phi = np.append(phi, getattr(self, k).flatten())
        return phi

    def phi2parms(self, phi):
        """ Unpack parameter values from a ndarray vector and use them to set the
        parameters of this module. """
        os = 0
        for k in self.fit_fields:
            s = getattr(self, k).shape
            setattr(self, k, phi[os:(os+np.prod(s))].reshape(s))
            os += np.prod(s)

    def evaluate(self, myfn):
        """ Execute the module's code and transform d_in into d_out. """
        self.d_out = copy.copy(self.d_in)
        args = {k: self.d_in[k] for k in self.input_signal_names}
        self.d_out[self.output_signal_name] = myfn(args)

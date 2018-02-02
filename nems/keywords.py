# This dict maps keywords to fragments of a modelspec
# TODO: added api tag for easy imports from nems.modules.api.
#       Maybe not the best approach? Just trying to get demo running for now.
# TODO: added some dummy phi values to fir and dexp, currently those
#       functions have all pos. args. so have to be specified.
defaults = {'wc1': {'fn': 'nems.modules.weight_channels.weight_channels',
                    'fn_kwargs': {},
                    'prior': [],
                    'phi': {}},
            'wc2': {'fn': 'nems.modules.weight_channels.weight_channels',
                    'fn_kwargs': {},
                    'prior': [],
                    'phi': {}},
            'fir10x1': {'fn': 'nems.modules.fir.fir_filter',
                        'fn_kwargs': {},
                        'prior': [],
                        'phi': {'coefficients': [[0.0, 0.0, 0.0]]}},
            'fir10x2': {'fn': 'nems.modules.fir.fir_filter',
                        'fn_kwargs': {},
                        'prior': [],
                        'phi': {'coefficients': [[0.0, 0.0, 0.0]]}},
            'dexp1': {'fn': 'nems.modules.nonlinearity.double_exponential',
                      'fn_kwargs': {},
                      'prior': [],
                      'phi': {'base': 1.0, 'amplitude': 1.0,
                              'shift': 1.0, 'kappa': 1.0}},
            'dexp2': {'fn': 'nems.modules.nonlinearity.double_exponential',
                      'fn_kwargs': {'base': 1.0, 'amplitude': 1.0,
                                    'shift': 1.0, 'kappa': 1.0},
                      'prior': [],
                      'phi': {'base': 1.0, 'amplitude': 1.0,
                              'shift': 1.0, 'kappa': 1.0}}}

# This dict maps keywords to fragments of a modelspec

defaults = {'wc40x1': {'fn': 'nems.modules.weight_channels.weight_channels',
                       'fn_kwargs': {'i': 'stim',
                                     'o': 'pred'},
                       'phi': {'coefficients': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]}},
            'fir10x1': {'fn': 'nems.modules.fir.fir_filter',
                        'fn_kwargs': {'i': 'pred',
                                      'o': 'pred'},
                        'phi': {'coefficients': [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]}},
            'dexp1': {'fn': 'nems.modules.nonlinearity.double_exponential',
                      'fn_kwargs': {'i': 'pred',
                                    'o': 'pred'},
                      'phi': {'base': 1.0, 'amplitude': 1.0,
                              'shift': 1.0, 'kappa': 1.0}}}

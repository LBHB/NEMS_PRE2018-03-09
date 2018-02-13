# This dict maps keywords to fragments of a modelspec

defaults = {'wc40x1': {'fn': 'nems.modules.weight_channels.weight_channels',
                       'fn_kwargs': {'i': 'stim',
                                     'o': 'pred'},
                       'phi': {'coefficients': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                 1.0, 1.0, 1.0, 1.0]]}},
            'fir10x1': {'fn': 'nems.modules.fir.fir_filter',
                        'fn_kwargs': {'i': 'pred',
                                      'o': 'pred'},
                        'phi': {'coefficients': [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                                                  0.0, 0.0, 0.0, 0.0, 0.0]]}},
            'dexp1': {'fn': 'nems.modules.nonlinearity.double_exponential',
                      'fn_kwargs': {'i': 'pred',
                                    'o': 'pred'},
                      'phi': {'amplitude': 2.0},
                      'prior': {'base': ('Normal', [0, 10])}
                      }}

# If not specified in the modelspec, these priors will be used
default_priors = {'nems.modules.fir.fir_filter':
                  {'coefficients': ('Normal', [[[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]],
                                               [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]])},
                  'nems.modules.nonlinearity.double_exponential':
                  {'base': ('Normal', [0, 1]),
                   'amplitude': ('HalfNormal', [0.5, 0.5]),
                   'shift': ('Normal', [0, 1]),
                   'kappa': ('HalfNormal', [0.5, 0.5])}}

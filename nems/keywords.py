# This dict maps keywords to fragments of a modelspec
import numpy as np

defaults = {}


def defkey(keyword, modulespec):
    '''
    Adds modulespec to the defaults keyword dictionary.
    A helper function so not every keyword mapping has to be in a single
    file and part of a very large single multiline dict.
    '''
    if keyword in defaults:
        raise ValueError("Keyword already defined! Choose another name.")
    defaults[keyword] = modulespec


def defkey_wc(n_inputs, n_outputs):
    name = 'wc{}x{}'.format(n_inputs, n_outputs)
    p_coefficients = {
        'mu': np.zeros((n_outputs, n_inputs)),
        'sd': np.ones((n_outputs, n_inputs)),
    }
    template = {
        'fn': 'nems.modules.weight_channels.weight_channels',
        'fn_kwargs': {'i': 'stim', 'o': 'pred'},
        'prior': {
            'coefficients': ('Normal', p_coefficients),
        }
    }
    return defkey(name, template)


defkey_wc(40, 1)
defkey_wc(18, 1)
defkey_wc(18, 2)


defkey('fir10x1',
       {'fn': 'nems.modules.fir.fir_filter',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'coefficients':
                  ('Normal', {'mu': [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              'sd': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})}})

defkey('fir15x1',
       {'fn': 'nems.modules.fir.fir_filter',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'coefficients':
                  ('Normal', {'mu': [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              'sd': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})}})

defkey('fir15x2',
       {'fn': 'nems.modules.fir.fir_filter',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'coefficients':
                  ('Normal', {'mu': [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                     [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                              'sd': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                                     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]})}})

defkey('lvl1',
       {'fn': 'nems.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'}})

defkey('dexp1',
       {'fn': 'nems.modules.nonlinearity.double_exponential',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mu': [0], 'sd': [1]}),
                  'amplitude': ('Normal', {'mu': [0.2], 'sd': [0.1]}),
                  'shift': ('Normal', {'mu': [0], 'sd': [1]}),
                  'kappa': ('Normal', {'mu': [0], 'sd': [0.1]})}})

defkey('qsig1',
       {'fn': 'nems.modules.nonlinearity.quick_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mu': [0.1], 'sd': [0.1]}),
                  'amplitude': ('Normal', {'mu': [0.7], 'sd': [0.5]}),
                  'shift': ('Normal', {'mu': [1.5], 'sd': [1.0]}),
                  'kappa': ('Normal', {'mu': [0.1], 'sd': [0.1]})}})

defkey('logsig1',
       {'fn': 'nems.modules.nonlinearity.logistic_sigmoid',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mu': [0], 'sd': [1]}),
                  'amplitude': ('Normal', {'mu': [0.2], 'sd': [1]}),
                  'shift': ('Normal', {'mu': [0], 'sd': [1]}),
                  'kappa': ('Normal', {'mu': [0], 'sd': [0.1]})}})

defkey('tanh1',
       {'fn': 'nems.modules.nonlinearity.tanh',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'},
        'prior': {'base': ('Normal', {'mu': [0], 'sd': [1]}),
                  'amplitude': ('Normal', {'mu': [0.2], 'sd': [1]}),
                  'shift': ('Normal', {'mu': [0], 'sd': [1]}),
                  'kappa': ('Normal', {'mu': [0], 'sd': [0.1]})}})

# This dict maps keywords to fragments of a modelspec

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


# Define keywords like this:
defkey('wc40x1',
       {'fn': 'nems.modules.weight_channels.weight_channels',
        'fn_kwargs': {'i': 'stim',
                      'o': 'pred'},
        'prior': {'coefficients':
                  ('Normal', {'mu': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0]],
                              'sd': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0]]})}})
    
defkey('wc18x1',
       {'fn': 'nems.modules.weight_channels.weight_channels',
        'fn_kwargs': {'i': 'stim',
                      'o': 'pred'},
        'phi':{'coefficients': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]},
        'prior': {'coefficients':
                  ('Normal', {'mu': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                              'sd': [[1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                      1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]})}})

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

defkey('lvl1',
       {'fn': 'nems.modules.levelshift.levelshift',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'}})

defkey('dexp1',
       {'fn': 'nems.modules.nonlinearity.double_exponential',
        'fn_kwargs': {'i': 'pred',
                      'o': 'pred'}})

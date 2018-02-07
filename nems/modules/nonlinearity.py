import numpy as np


def double_exponential(rec=None, i=None, o=None, base=None,
                       amplitude=None, shift=None, kappa=None):
    '''
    A double exponential applied to all channels of a single signal.
       rec        Recording object
       i          Input signal name
       o          Output signal name
       base       Y-axis height of the center of the sigmoid
       amplitude  Y-axis distance from ymax asymptote to ymin asymptote
       shift      Centerpoint of the sigmoid along x axis
       kappa      Sigmoid curvature (higher is...steeper? TODO)
    '''
    fn = lambda x : base + amplitude * np.exp(-np.exp(-kappa * (x - shift)))
    return rec.add_signal(rec[i].transform(fn, o))

import numpy as np


def double_exponential(rec, i, o, base, amplitude, shift, kappa):
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
    fn = lambda x : base + amplitude * np.exp(-np.exp(-np.exp(kappa) * (x - shift)))
    return [rec[i].transform(fn, o)]

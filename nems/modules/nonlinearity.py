from functools import partial
import numpy as np


def _double_exponential(x, base, amplitude, shift, kappa):
    return base + amplitude * np.exp(-np.exp(-kappa * (x - shift)))


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
    # fn = lambda x : base + amplitude * np.exp(-np.exp(-kappa * (x - shift)))
    # fn = partial(_double_exponential,
    #              base=base,
    #              amplitude=amplitude,
    #              shift=shift,
    #              kappa=kappa)
    return [rec[i].transform(_double_exponential,
                             (base, amplitude, shift, kappa)).rename(o)]

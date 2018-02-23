import cProfile
import numpy as np
import numexpr
from numpy import exp



def _logistic_sigmoid(x, base, amplitude, shift, kappa):
    ''' This "logistic" function only has a single negative exponent '''
    return base + amplitude * 1 / (1 + exp(-kappa * (x - shift)))

def logistic_sigmoid(rec, i, o, base, amplitude, shift, kappa):
    fn = lambda x : _logistic_sigmoid(x, base, amplitude, shift, kappa)
    return [rec[i].transform(fn, o)]



def _tanh(x, base, amplitude, shift, kappa):
    return base + (0.5 * amplitude) * (1 + np.tanh(kappa * (x - shift)))

def tanh(rec, i, o, base, amplitude, shift, kappa):
    fn = lambda x : _tanh(x, base, amplitude, shift, kappa)    
    return [rec[i].transform(fn, o)]


def _quick_sigmoid(x, base, amplitude, shift, kappa):
    y = kappa * (x - shift)
    return base + (0.5 * amplitude) * (1 + y / np.sqrt(1 + np.square(y)))

def quick_sigmoid(rec, i, o, base, amplitude, shift, kappa):
    fn = lambda x : _quick_sigmoid(x, base, amplitude, shift, kappa)
    return [rec[i].transform(fn, o)]


def _double_exponential(x, base, amplitude, shift, kappa):
    # Apparently, numpy is VERY slow at taking the exponent of a negative number
    # https://github.com/numpy/numpy/issues/8233
    # This is solved by using the MKL array. Uncomment this if trying to benchmark.
    # shifted_and_scaled = np.array(-1.0 * np.exp(kappa)) * (x - shift)
    # exp_of_shifted = np.exp(shifted_and_scaled)
    # neg_exp_of_shifted = -1.0 * exp_of_shifted
    # exponents = np.exp(neg_exp_of_shifted)
    # result = base + amplitude * exponents
    # return result
    #return numexpr.evaluate("base + amplitude * exp(-exp(-exp(kappa) * (x - shift)))")
    return base + amplitude * np.exp(-np.exp(-np.exp(kappa) * (x - shift)))

def double_exponential(rec, i, o, base, amplitude, shift, kappa):
    '''
    A double exponential applied to all channels of a single signal.
       rec        Recording object
       i          Input signal name
       o          Output signal name
       base       Y-axis height of the center of the sigmoid
       amplitude  Y-axis distance from ymax asymptote to ymin asymptote
       shift      Centerpoint of the sigmoid along x axis
       kappa      Sigmoid curvature. Larger numbers mean steeper slopes.
    We take exp(kappa) to ensure it is always positive.
    '''    
    fn = lambda x : _double_exponential(x, base, amplitude, shift, kappa)
    # fn = lambda x : _quick_sigmoid(x, base, amplitude, shift, kappa)
    # fn = lambda x : _tanh(x, base, amplitude, shift, kappa)
    # fn = lambda x : _logistic_sigmoid(x, base, amplitude, shift, kappa)
    return [rec[i].transform(fn, o)]



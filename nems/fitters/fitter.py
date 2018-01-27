import time
import numpy as np

import nems.fitters.termination_conditions as tc


def dummy_fitter(sigma, cost_fn, bounds=None, fixed=None):
    """just for testing that everything is connected, can remove later."""
    err = cost_fn(sigma=sigma)
    print("I did a 'fit'! err was: {0}".format(err))
    return sigma

def bit_less_dummy_fitter(sigma, cost_fn, bounds=None, fixed=None):
    # TODO: change stepinfo to class acting as thin wrapper around
    #       a dict? just to make expected attributes more explicit.
    #       subscripts in code should be identical, would only change
    #       from stepinfo = {} to stepinfo = StepInfo()
    #       (see my comments in .termination_conditions.py) --jacob 1-26-18
    stepinfo = {
            'num': 0,
            'err': cost_fn(sigma=sigma),
            'err_delta': np.inf,
            'start_time': time.time()
            }

    # fit loop
    stop_fit = lambda stepinfo: (tc.max_iterations_reached(stepinfo, 10)
                                 or tc.fit_time_exceeded(stepinfo, 30))
    while not stop_fit(stepinfo):
        print("sigma is now: {0}".format(sigma))
        sigma[0] = stepinfo['num']
        err = cost_fn(sigma=sigma)
        err_delta = err - stepinfo['err']
        stepinfo['err'] = err
        stepinfo['err_delta'] = err_delta
        stepinfo['num'] += 1
        print("Stepinfo is now: {}".format(stepinfo))

    return sigma
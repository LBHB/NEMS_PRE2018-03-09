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

def coordinate_descent(sigma, cost_fn, step_size=0.1, step_change=0.5,
                       tolerance=1e-7):
    stepinfo = {
            'num': 0,
            'err': cost_fn(sigma=sigma),
            'err_delta': np.inf,
            'start_time': time.time()
            }

    stop_fit = lambda stepinfo: (tc.error_non_decreasing(stepinfo, tolerance)
                                 or tc.max_iterations_reached(stepinfo, 1000))
    while not stop_fit(stepinfo):
        n_parameters = len(sigma)
        step_errors = np.zeros([n_parameters, 2])
        for i in range(0, n_parameters):
            # Try shifting each parameter both negatively and positively
            # proportional to step_size, and save both the new
            # sigma vectors and resulting cost_fn outputs
            this_sigma_pos = sigma.copy()
            this_sigma_neg = sigma.copy()
            this_sigma_pos[i] += this_sigma_pos[i]*step_size
            this_sigma_neg[i] -= this_sigma_neg[i]*step_size
            step_errors[i, 0] = cost_fn(sigma=this_sigma_pos)
            step_errors[i, 1] = cost_fn(sigma=this_sigma_neg)
        # Get index tuple for the lowest error that resulted,
        # and keep the corresponding sigma vector for the next iteration
        i_param, j_sign = np.unravel_index(
                                step_errors.argmin(), step_errors.shape
                                )
        # If j is 1, shift was negative
        if j_sign:
            sigma[i_param] -= sigma[i_param]*step_size
        else:
            sigma[i_param] += sigma[i_param]*step_size
        err = step_errors[i_param, j_sign]

        # update stepinfo
        err = cost_fn(sigma=sigma)
        stepinfo['num'] += 1
        stepinfo['err_delta'] = stepinfo['err'] - err
        stepinfo['err'] = err

        if stepinfo['err_delta'] < 0:
            print("Error got worse, reducing step size from: {0} to: {1}"
                  .format(step_size, step_size*step_change))
            step_size *= step_change

    return sigma
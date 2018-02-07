import numpy as np

import nems.fitters.termination_conditions as tc


def dummy_fitter(sigma, cost_fn, bounds=None, fixed=None):
    '''
    This fitter does not actually take meaningful steps; it merely
    varies the first element of the sigma vector to be equal to the step
    number. It is intended purely for testing and example purposes so
    that you can see how to re-use termination conditions to write
    your own fitter.
    '''
    # Define a stepinfo and termination condition function 'stop_fit'
    stepinfo, update_stepinfo = tc.create_stepinfo()
    stop_fit = lambda : (tc.error_non_decreasing(stepinfo, 1e-5) or
                         tc.max_iterations_reached(stepinfo, 1000))

    while not stop_fit():
        sigma[0] = stepinfo['stepnum']  # Take a fake step
        err = cost_fn(sigma=sigma)
        update_stepinfo(err=err, sigma=sigma)

    return sigma


def coordinate_descent(sigma, cost_fn, step_size=0.1, step_change=0.5,
                       tolerance=1e-7):
    stepinfo, update_stepinfo = tc.create_stepinfo()
    stop_fit = lambda : (tc.error_non_decreasing(stepinfo, tolerance) or
                         tc.max_iterations_reached(stepinfo, 1000))

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
        update_stepinfo(err=err)

        if stepinfo['err_delta'] < 0:
            print("Error got worse, reducing step size from: {0} to: {1}"
                  .format(step_size, step_size*step_change))
            step_size *= step_change

    return sigma

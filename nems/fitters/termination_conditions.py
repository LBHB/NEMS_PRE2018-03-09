"""TODO: Generic termination condition functions for fitters.
How much can we generalize here?

Goals as sketched out by pure functions mockup (and some of my interp.)_
    Termination condition functions should take in a step_info dict
    and return a Boolean, nothing else.
    If more than one condition should be checked, fitters can refer to
    more than one function to compose them.
    ex: A = error_nondecreasing(step_info)
        B = some_other_condition_function(step_info)
        C = ...etc

        if A and B and C:
            break

Proposed Conventions:
    Name of the function should describe the event that will cause the
    fitting loop to stop, and a return value of True should likewise
    indicate a need to stop.
    For example, 'error_non_decreasing' returns a value of True
    when, as the name implies, the error is no longer decreasing by
    an amount at least equal to the specified tolerance.
    As a result, fitters should generally refer to the step_condition
    in terms of: 'if termination_condition <is True>:  <stop fitting>'
    Using the reverse naming and return values should be avoided.
    For example: 'error_still_decreasing':  return False if err_delta < tol
    Which might read as:
        'if not error_still_decreasing() <is True>: <stop fitting>'

As for expected structure of step_info:

    Copied from pure functions mockup:

    stepinfo = {'num': 1, # Num of steps
                'err': cost_fn(spec), #
                'err_delta': None, # Change since last step
                'start_time': time.time()}

    The 4 above seem like obvious inclusions. Anything else?
    For now, going to say that all termination condition checkers
    should be able to rely on those 4 keys being present.

    TODO:
    Maybe this is a good case for a tiny class with
    subscriptability to make the expected keys more explicit?
    Can still add others, but this would at least ensure that no
    KeyErrors get thrown if a fitter isn't defining start_time,
    for example. Overall not much functional difference,
    could just as easily expect users to read the
    termination_condition docs and define the structure of
    the stepinfo dict there.

    i.e. along the lines of:
        class StepInfo:
            num = None
            err = None
            err_delta = None
            start_time = None

            def __init__(self, kwargs):
                for key, val in kwargs.items():
                    setattr(self, key, val)
            def __getitem__(self, key):
                return self.__dict__[key]
            def __setitem__(self, key, val):
                self.__dict__[key] = val

    --jacob 1-26-18

"""

import time
import numpy as np

def error_non_decreasing(stepinfo, tolerance=1e-5):
    """If stepinfo's 'err_delta' is less than tolerance, returns True."""
    # Using absolute value because fitters might be
    # defining delta as  err_i - err_i+1  or  err_i+1 - err_i
    if np.abs(stepinfo['err_delta']) < tolerance:
        return True
    else:
        return False

def fit_time_exceeded(stepinfo, max_time=600):
    """If stepinfo's 'start_time' is at least max_time seconds ago,
    returns True.

    """

    # time.time() and stepinfo['start_time'] are in units of seconds
    # by default.
    if (time.time() - stepinfo['start_time']) >= max_time:
        print("Maximum fit time exceeded: {}".format(max_time))
        return True
    else:
        return False

def max_iterations_reached(stepinfo, max_iter=10000):
    """If stepinfo's 'num' is greater than or equal to max_iter,
    returns True.

    """
    if stepinfo['num'] >= max_iter:
        print("Maximum iterations reached: {}".format(max_iter))
        return True
    else:
        return False

def target_err_reached(stepinfo, target=0.0):
    """If stepinfo's 'err' is less than or equal to target, returns True."""
    if stepinfo['err'] <= target:
        return True
    else:
        return False

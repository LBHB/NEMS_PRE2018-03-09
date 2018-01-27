import time

class Fitter:

    def fit(self, model, signals):
        raise NotImplementedError

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
    stepinfo = {}

    stepinfo['err'] = cost_fn(sigma=sigma)
    stepinfo['start_time'] = time.time()

    # fit loop
    count = 0
    # TODO: use a term cond func w/ SI
    termination_condition = lambda count: (count == 10)
    while not termination_condition(count):
        print("sigma is now: {0}".format(sigma))
        sigma[0] = count
        err = cost_fn(sigma=sigma)
        err_delta = err - stepinfo['err']
        stepinfo['err'] = err
        stepinfo['err_delta'] = err_delta
        count += 1
        stepinfo['num'] = count
        print("loop # {0}, stepinfo is now: {1}".format(count, stepinfo))

    return sigma
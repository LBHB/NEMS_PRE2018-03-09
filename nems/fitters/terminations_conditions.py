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
    i.e. along the lines of:
        class StepInfo:
            num = None
            err = None
            err_delta = None
            start_time = None

            def __getitem__(self, key):
                return self.__dict__[key]
            def __setitem__(self, key, val):
                self.__dict__[key] = val

    --jacob 1-26-18

"""



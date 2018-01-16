import logging
log = logging.getLogger(__name__)

from functools import partial

from scipy import optimize

from . import util
from . import cost_functions


def get_optimize_callback(n, function=None, printer=print):
    '''
    Create a stateful callback that can be used by `scipy.optimize`

    This uses a generator to maintain function state across calls. This allows
    us to track the number of times Scipy evaluates the function without the
    computational overhead of a Python class.

    Parameters
    ----------
    n : int
        Print message every n evaluations
    function : {callable, None)
        Function being optimized. If provided, the message will include the
        mean-squared error on that cycle.
    printer : callable
        Printer to use. If usign the logging system, you can pass `log.info` or
        `log.debug`. If you just want printed to the standard output, just use
        the default of `print`.

    Example
    -------
    >>> cb = get_optimize_cb(3)
    >>> cb()
    >>> cb()
    >>> cb()
    Iteration 3
    >>> cb()
    >>> cb()
    >>> cb()
    Iteration 6
    '''
    def optimize_cb(n, function):
        i = 0
        while True:
            phi = (yield)
            i += 1
            if i % n:
                if function is not None:
                    cost = function(phi)
                    print('Iteration {}: {}'.format(i*n, cost))
                else:
                    print('Iteration {}'.format(i*n))

    # Create the generator and initialize it
    cb = optimize_cb(n, function)
    next(cb)

    # THe `send` method on the generator object will receive the value (phi)
    # provided by scipy.optimize.
    return cb.send


def _fit_function(phi_vector, phi_template, model, signals, cost_function):
    '''
    Thin wrapper around the model evaluation system to facilitate Scipy fitters.
    Meant for internal use by `fit`.

    Parameters
    ----------
    phi_vector : 1d array
        Array of values for phi
    phi_template : list of dictionaries
        Required to help convert the vector for back to the list of dictionaries
        required by the module evaluation algorithms.
    model : instance of `nems.Model`
        Model to fit.
    signals : dictionary of arrays
        Data to fit.
    cost_function : callable
        Takes the output of the model and returns a single value (e.g., the mean
        squared error).

    Returns
    -------
    cost : float
        Value calculated by the cost function
    '''
    # Convert the vector back to the list of dicts required by Model.evaluate
    phi = util.vector_to_phi(phi_vector, phi_template)

    # Evaluate given phi and extract the pred/resp values
    result = model.evaluate(signals, phi)

    # Return the cost
    return cost_function(result)


def fit(model, signals, cost_function=None, phi_initialization='mean',
        phi_bounds=(0.01, 0.99), initial_phi=None, **optimize_kw):
    '''
    Fit model using standard Scipy optimization routines

    Parameters
    ----------
    model : instance of `nems.Model`
        Model to fit
    signals : dictionary of arrays
        Data to fit
    cost_function : callable
        Callable that accepts one argument (the output of `model.evaluate`) and
        returns the metric to be minimized. To maximize a metric, return the
        negative of that value.
    phi_initialization : {string, float}
        Method to compute starting values of phi for the fit.
    phi_bounds : tuple of two floats
        Percentiles (in fraction) of the prior to use for calculating
        coefficient boundaries. For continuous priors (e.g., Normal),
        percentiles of 0 and 1 will return -Inf and +Inf, respectively. Often
        you want to clip to a finite range of values to maximize optimization
        success. Try 0.01 through 0.99 to give you a fairly broad search range
        that covers 98% of the expected values for that coefficient.
    initial_phi : {None, list of dicts}
        This allows you to specify a set of starting points for phi for a subset
        of the modules rather than using the defaults generated by the phi
        initialization routine. This is typically only used by the iterative
        fitter.
    optimize_kw:
        Keyword arguments passed to `scipy.optimize`.

    Returns
    -------
    result : dictionary
        Keys include `phi`, the fitted value of phi and `result`, the object
        returned by scipy. Eventually this will be fixed to more directly
        reflect the desired output for NEMS.

    Example
    -------
    This is the simplest approach
    >>> signals = load_signals()
    >>> model = create_model('wcg02_fir15_dexp')
    >>> result = fit(model, signals)

    To print out a status update every 1000 function evaluations
    >>> signals = load_signals()
    >>> model = create_model('wcg02_fir15_dexp')

    >>> result = fit(model, signals, callback=callback)

    To specify your own cost function that uses `resp-all-cells` instead of
    `resp`
    >>> signals = load_signals()
    >>> model = create_model('wcg02_fir15_dexp')
    >>> cost_function = partial(cost_functions.mse, resp_name='resp-all-cells')
    >>> result = fit(model, signals, cost_function)
    '''
    # If not provided, set to the default mean-squared error
    if cost_function is None:
        cost_function = cost_functions.mse

    # Get list of priors for each module. Each prior tells us a bit about
    # reasonable values for the corresponding coefficient. We don't use the
    # prior distributions directly in a simple Scipy optimize routine; however,
    # the priors are still useful as they can be used to determine boundaries on
    # the coefficients.
    priors = model.get_priors(signals)

    # We can initialize the starting value of phi to the mean value or random
    # value of the distribution.
    phi = util.initialize_phi(priors, phi_initialization)

    # Copy initial values of phi over if specified. Typically used by iterative
    # fitting routines.
    if initial_phi is not None:
        n = len(initial_phi)
        phi[:n] = initial_phi

    # Construct the boundaries for phi
    phi_lower = util.phi_percentile(priors, phi_bounds[0])
    phi_upper = util.phi_percentile(priors, phi_bounds[1])
    phi_lower_vector = util.phi_to_vector(phi_lower)
    phi_upper_vector = util.phi_to_vector(phi_upper)
    bounds = list(zip(phi_lower_vector, phi_upper_vector))

    # Build our fit function. Use functools.partial to freeze the arguments.
    # TODO: do some testing to see whether functools.partial should be
    # superseded in lieu of having scipy.optimize provide the list of vectors.
    # In theory, best to build a complete symbolic graph of the computations
    # that is then composited into a single, fast function. Test this
    # eventually.
    function = partial(_fit_function, phi_template=phi,
                       model=model, signals=signals,
                       cost_function=cost_function)

    # Convert phi to the format required by the fitter
    phi_vector = util.phi_to_vector(phi)

    # Run the fit
    result = optimize.minimize(function, phi_vector, bounds=bounds,
                               **optimize_kw)

    # Convert phi back to required format
    fitted_phi = util.vector_to_phi(result.x, phi)

    return {
        'phi': fitted_phi,
        'result': result,
    }


def iterative_fit(model, signals, cost_function=None, phi_initialization='mean',
                  phi_bounds=(0.01, 0.99), initial_phi=None, **optimize_kw):

    '''
    Iterative version of the fit function. See `fit` for explanation of
    parameters and return value (note that `initial_phi` has different behavior
    in this function as described below).

    Parameters
    ----------
    initial_phi : {None, list of dictionaries}
        If provided, these modules will be skipped and fitting will begin at the
        first module that does not have an entry in initial_phi. Note that this
        means that you cannot skip modules (i.e., initial_phi must have entries
        for every module starting from the first one in the model all the way up
        to the last module you want to specify `initial_phi` for).

    This supersedes the mini-fitting routines in the old NEMS codebase.
    '''
    fitted_phi = [] if initial_phi is None else initial_phi.copy()
    start = len(fitted_phi)

    for i in range(start, model.n_modules):
        log.debug('Fitting up to module number %d', i)

        # Extract subset of the model to be fit
        model_subset = model.iget_subset(ub=i+1)

        # Quick check to see whether the current module has any coefficients
        # that need to be fit. If not, skip ahead to the next module.
        priors = model_subset.get_priors(signals)
        if len(priors[-1]) == 0:
            continue

        # Use the fitted values of phi from the previous iterative fit. Note
        # that for the very first fit, `fitted_phi` is a dictionary of length 0.
        result = fit(model, signals, cost_function, phi_initialization,
                     phi_bounds, initial_phi=fitted_phi, **optimize_kw)

        # Extract the fitted value of phi and save this as the starting point
        # for the next module in the stack.
        fitted_phi = result['phi']

    # Return result from last iteration.
    return result

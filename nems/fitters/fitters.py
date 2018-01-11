#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import logging
log = logging.getLogger(__name__)

import os
from time import time
import scipy as sp
import numpy as np
import skopt
import nems.db as nd

def phi_to_vector(phi):
    '''
    Convert a list of dictionaries where the values are scalars or array-like
    to a single vector.

    This is a helper function for fitters that use scipy.optimize. The scipy
    optimizers require phi to be a single vector; however, it's more intuitive
    for us to return a list of dictionaries (where each dictionary in the list
    contains the values to be fitted.

    >>> phi = [{'baseline': 0, 'coefs': [32, 41]}, {}, {'a': 1, 'b': 15.1}]
    >>> phi_to_vector(phi)
    [0, 32, 41, 1, 15.1]

    >>> phi = [{'coefs': [[1, 2], [3, 4], [5, 6]]}, {'a': 32}]
    >>> phi_to_vector(phi)
    [1, 2, 3, 4, 5, 6, 32]
    '''
    vector = []
    for p in phi:
        for k in sorted(p.keys()):
            value = p[k]
            if np.isscalar(value):
                vector.append(value)
            else:
                flattened_value = np.asanyarray(value).ravel()
                vector.extend(flattened_value)
    return vector


def vector_to_phi(vector, phi_template):
    '''
    Convert vector back to a list of dictionaries given a template for phi

    >>> phi_template = [{'baseline': 0, 'coefs': [0, 0]}, {}, {'a': 0, 'b': 0}]
    >>> vector = [0, 32, 41, 1, 15.1]
    >>> vector_to_phi(vector, phi_template)
    [{'baseline': 0, 'coefs': array([32, 41])}, {}, {'a': 1, 'b': 15.1}]

    >>> phi_template = [{'coefs': [[0, 0], [0, 0], [0, 0]]}, {'a': 0}]
    >>> vector = [1, 2, 3, 4, 5, 6, 32]
    >>> vector_to_phi(vector, phi_template)
    [{'coefs': array([[1, 2], [3, 4], [5, 6]])}, {'a': 32}]
    '''
    # TODO: move this to a unit test instead?
    offset = 0
    phi = []
    for p_template in phi_template:
        p = {}
        for k in sorted(p_template.keys()):
            value_template = p_template[k]
            if np.isscalar(value_template):
                value = vector[offset]
                offset += 1
            else:
                value_template = np.asarray(value_template)
                size = value_template.size
                value = np.asarray(vector[offset:offset+size])
                value.shape = value_template.shape
                offset += size
            p[k] = value
        phi.append(p)
    return phi


class nems_fitter:
    """nems_fitter

    Generic NEMS fitter object

    """
    # Initial values of phi. This will be kept as the template to properly
    # reconstruct phi following each iteration of the eval.
    phi0 = None

    name = 'default'
    counter = 0
    fit_modules = []
    tolerance = 0.0001
    stack = None

    def __init__(self, parent, fit_modules=None, **xargs):
        self.stack = parent
        # figure out which modules have free parameters, if fit modules not
        # specified
        if not fit_modules:
            self.fit_modules = []
            for idx, m in enumerate(self.stack.modules):
                if m.fit_fields:
                    self.fit_modules.append(idx)
        else:
            self.fit_modules = fit_modules
        self.my_init(**xargs)

    def my_init(self, **xargs):
        pass

    # create fitter, this should be turned into an object in the nems_fitters
    # libarry
    def test_cost(self, phi):
        self.stack.modules[2].phi2parms(phi)
        self.stack.evaluate(1)
        self.counter += 1
        if self.counter % 100 == 0:
            log.info('Eval #{0}. MSE={1}'.format(
                self.counter, self.stack.error()))
        return self.stack.error()

    def do_fit(self):
        # run the fitter
        self.counter = 0
        # pull out current phi as initial conditions
        self.phi0 = self.stack.get_phi()
        vector = phi_to_vector(self.phi0)
        phi = sp.optimize.fmin(self.test_cost, self.phi0, maxiter=1000)
        return phi

    def tick_queue(self):
        if 'QUEUEID' in os.environ:
            queueid = os.environ['QUEUEID']
            nd.update_job_tick(queueid)


class basic_min(nems_fitter):
    """
    The basic fitting routine used to fit a model. This function defines a cost
    function that evaluates the moduels being fit using the current parameters
    and outputs the current mean square error (mse). The cost function is
    evaulated by the scipy optimize.minimize routine, which seeks to minimize
    the mse by changing the function parameters.

    Scipy optimize.minimize is set to use the minimization algorithm 'L-BFGS-B'
    as a default, as this is memory efficient for large parameter counts and
    seems to produce good results. However, there are many other potential
    algorithms detailed in the documentation for optimize.minimize

    """

    name = 'basic_min'
    maxit = 50000
    routine = 'L-BFGS-B'

    def my_init(self, routine='L-BFGS-B', maxit=50000, tolerance=1e-7):
        """
        Initializes the fitter.

        routine: the algorithm that scipy.optimize.minimize should use.
                L-BFGS-B and SLSQP tend to work very well, while Nelder-Mead,
                Powell, and BFGS work, but not as well. The documentation for
                scipy.optimize.minimize has more details on algorithms that can
                be used.
        maxit: maximum number of iterations for the fitter to use. Different
                than number of function evaluations.
        tolerance: the "accuracy" to which the cost function is fit. E.g.,
                if we have a tolerance of 0.001, the fitter will fit until the
                value of the cost function is stable at the third decimal place.

        """

        log.info("initializing basic_min")
        self.maxit = maxit
        self.routine = routine
        self.tolerance = tolerance

    def cost_fn(self, vector):
        """
        The cost function is the function that the fitting algorithm minimizes
        by changing the module parameters. This function takes in the vector of
        model parameters phi, applies the parameters in the fitted vector to
        the model using fit_to_phi, then calculates and returns the loss, here
        called error (usually mean squared error, but sometimes other functions
        such as huber loss).
        """

        phi = vector_to_phi(vector, self.phi0)
        self.stack.set_phi(phi)
        self.stack.evaluate(self.fit_modules[0])
        err = self.stack.error()
        self.counter += 1
        #if (self.counter % 200 == 0) and (self.counter < 1000):
            #log.debug("Eval # %d, phi vector is now: \n%s",
            #          self.counter, str(vector))
        if self.counter % 1000 == 0:
            log.info('Eval # %d', self.counter)
            log.info('Error=%.02f', err)
            #log.debug("Eval # %d, phi vector is now: \n%s",
            #          self.counter, str(vector))
            self.tick_queue()  # Update the progress indicator
        return(err)

    def do_fit(self):
        """
        The function to call to actually perform the fitting. The first few lines
        in this function set up some options specific to scipy.optimize.minimize.
        Below these, we have the core components of the nems_fitter object. First,
        we take the initial values of the model parameters and concatenate them
        all into a single vector, phi0, to pass to the fitting function. Counter
        is just a counter that keeps track of how many times cost_fn has been
        evaluated. The cost function and initial parameter values phi0 are then
        fed into the fit function, along with whatever options are needed for
        the specific fit function.

        Since cost_fn updates the mdoel parameters every time it is evaluated, we
        don't need to do anything else other than retun the final error (with the
        exception of simulated annealing).
        """

        opt = dict.fromkeys(['maxiter'])
        opt['maxiter'] = int(self.maxit)
        if self.routine == 'L-BFGS-B':
            opt['eps'] = 1e-7
        # TODO: Are there any modules with parameters that need constraints?
        #       Could add constraints per-module then extract
        #       similar to get_phi
        cons = ()

        # Initial guess = vector of current module parameter values
        self.phi0 = self.stack.get_phi(self.fit_modules)
        self.counter = 0
        vector = phi_to_vector(self.phi0)
        log.info("basic_min: phi0 initialized (fitting {0} parameters)"
                 .format(len(vector)))
        # TODO: Currently phi0 is almost all zeroes. Is this intended,
        #       or have initial guesses just not been added yet?
        log.debug("phi0 vector: \n{0}".format(vector))

        start = time()
        res = sp.optimize.minimize(
                self.cost_fn, vector, method=self.routine,
                constraints=cons, options=opt, tol=self.tolerance
                )
        end = time()
        elapsed = end-start
        log.debug("Minimization terminated\n"
                  "on eval #: {0}\n"
                  "after {1} seconds.\n"
                  "Success: {2}.\n"
                  "Reason: {3}\n"
                  "Optimized vector: {4}\n"
                  .format(self.counter, elapsed, res.success, res.message,
                          res.x))
        # stack.modules[-1] should be a metrics/error module,
        log.info("Final {0}: {1}\n"
                 .format(self.stack.modules[-1].name, self.stack.error()))
        return(self.stack.error())


class anneal_min(nems_fitter):
    """ A simulated annealing method to find the ~global~ minimum of your
    parameters.

    This fitter uses scipy.optimize.basinhopping, which is scipy's built-in
    annealing routine. Essentially, this routine uses scipy.optimize.minimize
    to minimize the function, then randomly perturbs the function and
    reminimizes it. It will continue this procedure until either the maximum
    number of iterations had been exceed or the minimum remains constant for
    a specified number of iterations.

    anneal_iter = number of annealing iterations to perform
    stop = number of iterations after which to stop annealing if global min
         remains the same
    up_int = update step size every up_int iterations
    maxiter = maximum iterations for each round of minimization
    tolerance = tolerance for each round of minimization
    min_method = method used for each round of minimization. 'L-BFGS-B'
                 works well
    bounds should be [(xmin,xmax),(ymin,ymax),(zmin,zmax),etc]

    WARNING: this fitter takes a ~~long~~ time. It is usually better to
    try basic_min first, and then use this method if basic_min fails.

    Also, note that since basinhopping (at least as implemented here) uses
    random jumps, the results my not be exactly the same every time, and the
    annealing may take a different number of iterations each time it is called
    @author: shofer, 30 June 2017

    Further note: this is currently set up to take small jumps, as might be
    useful for fitting FIR filters or small nonlinearities. To use this fitter
    effectively, the "expected" value of the coefficients must be taken into
    account.
    --njs, 5 July 2017

    """

    name = 'anneal_min'
    anneal_iter = 100
    stop = 5
    up_int = 10
    min_method = 'L-BFGS-B'
    maxiter = 10000
    tolerance = 0.01

    def my_init(self, min_method='L-BFGS-B', anneal_iter=100, stop=5,
                maxiter=10000, up_int=10, bounds=None,
                temp=0.01, stepsize=0.01, verb=False):
        log.info("initializing anneal_min")
        self.anneal_iter = anneal_iter
        self.min_method = min_method
        self.stop = stop
        self.maxiter = 10000
        self.bounds = bounds
        self.up_int = up_int
        self.temp = temp
        self.step = stepsize
        self.verb = verb

    def cost_fn(self, vector):
        phi = vector_to_phi(vector, self.phi0)
        self.stack.set_phi(phi)
        self.stack.evaluate(self.fit_modules[0])
        err = self.stack.error()
        self.counter += 1
        #if (self.counter % 200 == 0) and (self.counter < 1000):
            #log.debug("Eval # %d, phi vector is now: \n%s",
            #          self.counter, str(vector))
        if self.counter % 1000 == 0:
            log.info('Eval #' + str(self.counter))
            log.info('Error=' + str(err))
            #log.debug("Eval # %d, phi vector is now: \n%s",
            #          self.counter, str(vector))
        return(err)

    def do_fit(self):
        opt = dict.fromkeys(['maxiter'])
        opt['maxiter'] = int(self.maxiter)
        opt['eps'] = 1e-7
        min_kwargs = dict(
                method=self.min_method, tol=self.tolerance,
                bounds=self.bounds, options=opt,
                )
        self.phi0 = self.stack.get_phi(self.fit_modules)
        vector = phi_to_vector(self.phi0)
        self.counter = 0
        log.info("anneal_min: phi0 intialized (fitting {0} parameters)"
                 .format(len(vector)))
        log.debug("phi0 vector: \n{0}".format(vector))
        log.debug("maxiter: {0}".format(opt['maxiter']))

        start = time()
        opt_res = sp.optimize.basinhopping(
                self.cost_fn, vector, niter=self.anneal_iter,
                T=self.temp, stepsize=self.step, minimizer_kwargs=min_kwargs,
                interval=self.up_int, disp=self.verb, niter_success=self.stop
                )
        end = time()
        elapsed = end-start
        log.debug("Minimization terminated\n"
                  "on eval #: {0}\n"
                  "after {1} seconds.\n"
                  "Reason: {2}\n"
                  .format(self.counter, elapsed, opt_res.message))
        phi_final = opt_res.lowest_optimization_result.x
        self.cost_fn(phi_final)
        log.info("Final {0}: {1}\n"
                 .format(self.stack.modules[-1].name, self.stack.error()))
        return(self.stack.error())


class SkoptMin(nems_fitter):
    """Base class for Scikit-Optimize fit routines.

    Fits model parameters using Scikit-Optmize's gp_minimize.
    TODO: Finish this doc.

    """

    name = 'skopt_min'
    n_calls = 100
    maxit = 10000
    routine = 'skopt.gp_minimize'

    def my_init(self, dims=None, ncalls=100, maxit=10000):
        log.info("initializing scikit-optimize minimizer")
        self.n_calls = ncalls
        self.maxit = maxit
        self.dims = dims
        # Can pull after fitting to use some of skopts special
        # plotting routines, like plot_convergence
        self.fit_result = None

    def cost_fn(self, vector):
        phi = vector_to_phi(vector, self.phi0)
        self.stack.set_phi(phi)
        self.stack.evaluate(self.fit_modules[0])
        mse = self.stack.error()
        self.counter += 1
        if self.counter % 10 == 0:
            log.info('Eval %d', self.counter)
            log.info('MSE = %.02f', mse)
            #log.debug('Vector is now: %s', str(vector))

        return(mse)

    def min_func(self):
        result = skopt.gp_minimize(
                    func=self.cost_fn, dimensions=self.dims,
                    base_estimator=None, n_calls=100, n_random_starts=10,
                    acq_func='gp_hedge', acq_optimizer='auto', x0=self.x0,
                    y0=self.y0, random_state=None,
                    )
        return result

    def do_fit(self):
        # get initial guess at parameters
        self.phi0 = self.stack.get_phi(self.fit_modules)
        self.x0 = phi_to_vector(self.phi0)
        # evaluate error at initial guess
        self.y0 = self.cost_fn(self.x0)
        # figure out bounds for parameters
        if self.dims:
            pass
        else:
            # TODO: really need better constraints to use for these.
            #       Ideally, some distribution specified by the module,
            #       so that constraints can be set to +/- some number of
            #       standard deviations away from the mean.
            self.dims = []
            for param in self.x0:
                if param == 0:
                    cons = (-100, 100)
                else:
                    cons = (param-3*abs(param), param+3*abs(param))
                self.dims.append(cons)
            #log.debug("Dims ended up being: %s", str(self.dims))

        self.counter = 0
        log.info("SkoptMin: phi0 intialized (fitting %d parameters)",
                 len(self.x0))
        log.info("maxiter: %d", self.maxit)
        log.debug("Intial vector (x0) is: {0}".format(self.x0))
        self.fit_result = self.min_func()

        phi = vector_to_phi(self.fit_result.x, self.phi0)
        self.stack.set_phi(phi)
        self.stack.evaluate(self.fit_modules[0])
        log.info("Final MSE: {0}".format(self.stack.error()))
        log.debug("Optimized vector: {0}".format(self.fit_result.x))

        return(self.stack.error())


class SkoptForestMin(SkoptMin):
    """Fits model parameters using Scikit-Optimize's forest_minimize."""

    def min_func(self):
        result = skopt.forest_minimize(
                func=self.cost_fn, dimensions=self.dims, base_estimator="RF",
                n_calls=self.n_calls, n_random_starts=10, acq_func="EI",
                x0=self.x0, y0=self.y0, n_points=self.maxit, xi=self.tolerance,
                )
        return result

class SkoptGbrtMin(SkoptMin):
    """Fits model parameters using Scikit-Optimize's gprt_minimize."""

    def min_func(self):
        result = skopt.gbrt_minimize(
                func=self.cost_fn, dimensions=self.dims, base_estimator=None,
                n_calls=self.n_calls, n_random_starts=10, acq_func="EI",
                acq_optimizer="auto", x0=self.x0, y0=self.y0, xi=self.tolerance,
                kappa=0.001,
                )
        return result


class CoordinateDescent(nems_fitter):
    """
    coordinate descent - step one parameter at a time

    """

    name = 'CoordinateDescent'
    maxit = 1000
    tolerance = 0.000001
    step_init = 1.0
    step_change = 0.5
    step_min = 1e-7
    mult_step = False # steps by +/- step*value instead of +/- step
    dynamic_step_weight = False # weight steps based on prior err improvement

    # TODO: pseudo_cache'd results don't match their non-cached counterparts,
    #       so either something is wrong with that code or module evals aren't
    #       returning consistent output for same input and params.
    # TODO: figure out unit test for some module evals to see if they are
    #       returning random-ish output, unless that's desired behavior
    #       (in which case pseudo-cache concept won't work as intended).
    pseudo_cache = False  # skip repeat module evals
    # TODO: Anneal code "works," but doesn't help fit performance at all.
    anneal = 0  # of times to randomize initial inputs
    max_matches = 10  # stop anneal if error doesn't change
    # scales size of interval (uniform) or magnitude of standard deviation
    # (gaussian) for random values
    randomize_factor = 0.5
    random_type = 'gaussian' # can be uniform or gaussian

    # NOTES: Possible improvements?
    #        Fix anneal and pseudo-cache code (or abandon those approaches).
    #        Branching version? Instead of deciding on best improvement
    #            after one change to each param, do n number of steps for
    #            every option (similar to increasing number of look-aheads
    #            for game AI).
    #        Ex: for 4 params, increase and decrease each by step size.
    #            Then, repeat for each of those 8 results n number of times
    #            (until there are (2*#params)^n paths that were taken).
    #            Pick lowest error that results, then repeat.
    #        Pros: Includes any results that would come from current
    #            algorithm by definition, but also adds options that
    #            'greedy' CD wouldn't find.
    #        Cons: Would get very slow for even a low n
    #            (more than 3 or 4 would probably be unreasonable).
    #            Code would also be much more complicated since would need
    #            to eliminate redunant paths (i.e. no reason to increase
    #            a param on one step of a branch and then decrease it
    #            on the next step).

    def my_init(self, tolerance=0.000001, maxit=1000,
                step_init=1.0, mult_step=True, dynamic_step_weight=False,
                pseudo_cache=False, anneal=0, max_matches=10,
                randomize_factor=0.5, random_type='gaussian'):
        log.info("Initializing Coordinate Descent fitter.")
        self.maxit = maxit
        self.tolerance = tolerance
        self.step_init = step_init
        self.mult_step = mult_step
        self.dynamic_step_weight = dynamic_step_weight
        self.pseudo_cache = pseudo_cache
        self.anneal = anneal
        if self.anneal:
            self.do_anneal = True
        else:
            self.do_anneal = False
        self.random_type = random_type

    def cost_fn(self, vector):
        # If pseudo_cache is enabled, check to see if stack should
        # skip eval on some modules.
        first_changed_mod = 0
        if self.pseudo_cache:
            first_changed_mod = self._find_first_change(vector)

        phi = vector_to_phi(vector, self.phi0)
        self.stack.set_phi(phi)
        # If first_changed_mod ended up going past end of list, nothing
        # changed so don't need to re-eval at all
        # (This shouldn't happen during normal fitting)
        if not (first_changed_mod >= len(self.fit_modules)):
            #log.debug("First module with parameter changes was at idx: %d",
            #          self.fit_modules[first_changed_mod])
            self.stack.evaluate(self.fit_modules[first_changed_mod])
        else:
            log.debug("No changes to parameters detected since"
                      "last call to cost function, skipping eval."
                      "This should only happen if step size changed"
                      "due to insufficient error improvement.")
        mse = self.stack.error()

        return(mse)

    def _find_first_change(self, this_vec):
        """Before resetting stack phi, check parameters against previous set.

        For each module in the stack, if parameters didn't change,
        skip that module in eval unless it comes after a changed module.
        ex: if last vector was [0,0,1,1,0]
            and new vector is  [0,0,1,2,0],
            and each module takes 1 parameter so that these vectors
            correspond to 5 modules, then only the 4th and 5th module
            need to be re-evaluated.
        TODO: This assumes the modules don't have any random outputs,
              which I'm not 100% sure on. So need to double check that.
        TODO: Performance starts dropping compared to non-cached on
              more complicated models, so something isn't quite right with
              this yet.

        """

        last_phi = self.stack.get_phi()
        last_vec = phi_to_vector(last_phi)
        fit_mods_refs = [self.stack.modules[m] for m in self.fit_modules]
        parm_lens = []

        for i, m in enumerate(fit_mods_refs):
            mod_phi = m.get_phi()
            mod_vec = []
            for a in mod_phi:
                value = getattr(m, a)
                if np.isscalar(value):
                    mod_vec.append(value)
                else:
                    flattened_value = np.asanyarray(value).ravel()
                    mod_vec.extend(flattened_value)
            parm_lens.append(len(mod_vec))

        first_changed_mod = 0
        v_idx = 0
        changed = False
        for mod in parm_lens:
            # Check if every parameter for current module is the same
            for i in range(0, mod):
                if not (last_vec[v_idx] == this_vec[v_idx]):
                    changed = True
                    break
                v_idx += 1
            # If so, don't need to re-evaluate this module
            # unless an earlier module in stack was changed.
            if changed:
                break
            first_changed_mod += 1

        return first_changed_mod

    def _annealed_fit(self):
        """Run do_fit on initial parameters as normal, but also start from
        number of additional inputs equal to anneal count - 1 such that total
        number of fits equals self.anneal.

        Uses self.randomize_factor to calculate range for each parameter that
        random values should be picked from.
        ex: if param = 0.01 and randomize_factor = 10, then random values
            will be chosen between (0.01 - 10*0.1) and (0.01 + 10*0.01).
        Special case for param = 0: range between (0-1) and (0+1)
        TODO: Better way to handle this case?

        """

        log.info("Beginning annealed fit for Coordinate Descent fitter.\n"
                 "Number of anneals: %d.\n"
                 "Randomization factor: %d.",
                 self.anneal, self.randomize_factor)
        # Get initial parameters from stack and add them as first entry
        # in list of vectors
        self.phi0 = self.stack.get_phi()
        x0 = phi_to_vector(self.phi0)
        x_list = [x0]
        # Calculate intervals to choose random values from for each
        # parameter in x.
        random_starts = self.anneal-1
        param_intervals = [self._get_interval(p) for p in x0]
        param_distribs = [self._get_distribution(p) for p in x0]

        # Assemble list of randomized vectors
        for i in range(0, random_starts):
            x = x0.copy()
            for i, p in enumerate(x):
                # uniform random
                if self.random_type == 'uniform':
                    interval = param_intervals[i]
                    x[i] = np.random.choice(interval)
                # gaussian random
                elif self.random_type == 'gaussian':
                    mu, sigma = param_distribs[i]
                    randn = (sigma * np.random.randn() + mu)
                    x[i] = randn
                else:
                    log.warning("No random_type specified, or specification"
                                "was invalid. Using initial param value for"
                                "i=%d", i)
                    x[i] = x0[i]
            x_list.append(x)
        #log.debug("Random vectors assembled: %s", str(x_list))

        # Run self.do_fit() once for each anneal count, starting with
        # x0 followed by the randomized vectors.
        scores = []
        min_idx = 0
        min_score = 10e10
        loops_finished = 0
        for i in range(0, self.anneal):
            x = x_list[i]
            this_phi = vector_to_phi(x, self.phi0)
            self.stack.set_phi(this_phi)
            log.info("Anneal # %d, phi vector is: %s", i, str(x))
            this_score = self.do_fit()
            scores.append((this_score, x))

            # Update minimum score and its index in scores
            if this_score < min_score:
                min_score = this_score
                min_idx = i
            loops_finished += 1

            # If the last n scores were the same, stop annealing
            n = self.max_matches
            if len(scores) >= n:
                match_checks = [abs((round(s[0]-this_score)) > self.tolerance)
                                for s in scores[-n:]]
                log.debug("match_checks as intended? should be bool list: %s",
                          str(match_checks))
                if all(match_checks):
                    log.info("Last %d scores were the same, stopping fit.", n)
                    break

        log.info("Annealing completed after %d iterations.\n"
                  "Minimum score was: %.09f.\n"
                  "From loop: %d.",
                  loops_finished, min_score, min_idx)
        min_x = scores[min_idx][1]
        score_list = [s[0] for s in scores]
        log.debug("Optimized parameters were: %s", str(min_x))
        log.debug("All scores obtained were: %s", str(score_list))
        min_phi = vector_to_phi(min_x, self.phi0)
        self.stack.set_phi(min_phi)
        self.stack.evaluate(self.fit_modules[0])

        return min_score

    def _get_interval(self, p):
        """Returns an array of n = 10*self.randomize_factor
        uniform values surrounding p.

        """

        if p == 0:
            lower_bound = -1
            upper_bound = 1
        else:
            lower_bound = p-(p*self.randomize_factor)
            upper_bound = p+(p*self.randomize_factor)
        interval = np.linspace(lower_bound, upper_bound,
                               num=100*self.anneal)
        return interval

    def _get_distribution(self, p):
        """Returns a pair of mu and sigma values for generating a normal
        distribution around the value p.

        """

        mu = p
        if p == 0:
            # TODO: What about params with larger values that started
            #       at 0?
            sigma = self.randomize_factor
        else:
            sigma = p*self.randomize_factor

        return mu, sigma

    def _termination_condition(self, s_delta, step_size, n):
        stop = False

        if n >= self.maxit:
            stop = True
        if step_size <= self.step_min:
            stop = True
        if s_delta > 0 and s_delta < self.tolerance:
            # Normally want to stop at this point, but if step
            # size hasn't reached minimum then try that first.
            if step_size > self.step_min:
                stop = False
            else:
                stop = True

        return stop

    def do_fit(self):
        if self.do_anneal:
            self.do_anneal = False
            s = self._annealed_fit()
            return s

        self.phi0 = self.stack.get_phi()
        # TODO: Do we need this? Not sure what this corresponds to in matlab.
#        if ~options.Elitism
#            options.EliteParams = n_params;
#            options.EliteSteps = 1;
#        end

        n = 1   # step counter
        x = phi_to_vector(self.phi0)  # current phi
        n_params = len(x)
        log.debug("Initial parameters: %s", str(x))
        x_save = x.copy()     # last updated phi
        s = self.cost_fn(x)   # current score
        # Improvement of score over the previous step
        s_new = np.zeros([n_params, 2])
        # Weights will update after each iteration if dynamic_step_weight on.
        # Multiplies step amount by the inverse of the ratio of change
        # in error to step size, so that params with small improvements
        # try bigger steps and vise versa.
        param_weights = np.ones([n_params, 2])
        s_delta = np.inf     # Improvement of score over the previous step
        step_size = self.step_init  # Starting step size.
        log.info("{0}: phi0 intialized (start error={1}, {2} parameters)"
                 .format(self.name, s, n_params))
        log.info("starting CD: step size: {0:.9f} tolerance: {1:.9f}"
                 .format(step_size, self.tolerance))

        # Iterate until change in error is smaller than tolerance,
        # but stop if max iterations exceeded or minimum step size reached.
        start = time()
        while not self._termination_condition(s_delta, step_size, n):
            for ii in range(0, n_params):
                # Alternate adding and subtracting stepsize from each param,
                # then run cost function on the new x and store
                # the result in s_new. Reset x in between +/- so that only
                # one param at a time is changed.
                # ex: if step size is 1, and x started as [0,0,0],
                #     then after 3 loops s_new will store:
                #     ([cf([1,0,0]), cf([0,1,0]) , cf([0,0,1])],
                #      [cf([-1,0,0]), cf([0,-1,0]), cf([0,0,-1])]),
                #     where cf abbreviates self.cost_fun
                for ss in [0, 1]:

                    x[:] = x_save[:]
                    change = x[ii]*step_size*param_weights[ii,ss]
                    if ss == 0:
                        if self.mult_step:
                            x[ii] += change
                        else:
                            x[ii] += step_size
                    else:
                        if self.mult_step:
                            x[ii] -= change
                        else:
                            x[ii] -= step_size
                    err = self.cost_fn(x)
                    s_new[ii, ss] = err

                    improvement = s-err
                    if improvement <= 0:
                        improvement_rate = 1
                    else:
                        improvement_rate = improvement/step_size
                    #log.debug("err was: %.09f", err)
                    #log.debug("improvement_rate was: %.09f", improvement_rate)
                    if self.dynamic_step_weight:
                        param_weights[ii, ss] = 1/improvement_rate
                        #log.debug("parameter weights are now: %s",
                        #          str(param_weights))

            # get the array index in s_new corresponding to the smallest
            # error returned  by self.cost_fun on the stepped x values.
            x_opt = np.unravel_index(s_new.argmin(), s_new.shape)
            param_idx, sign_idx = x_opt
            s_delta = s - s_new[x_opt]

            if s_delta < 0:
                step_size = step_size * self.step_change
                log.debug("%d: Backwards (delta=%.09f), "
                         "adjusting step size to %.09f",
                         n, s_delta, step_size)

            elif s_delta < self.tolerance:
                step_size = step_size * self.step_change
                log.debug("%d: Error improvement too small (delta=%.09f)\n"
                         "Old score: %.09f\n"
                         "New score: %.09f",
                         n, s_delta, s, s_new[x_opt])
                log.debug("adjusting step size to %.09f", step_size)

            # sign_idx 0 means positive change was better
            # sign_idx 1 means negative change was better
            elif sign_idx:
                x_save[param_idx] -= x_save[param_idx]*step_size
                log.debug("%d: best step=(%d,%d) error=%.06f, delta=%.09f",
                         n, param_idx, sign_idx, s_new[x_opt], s_delta)
            else:
                x_save[param_idx] += x_save[param_idx]*step_size
                log.debug("%d: best step=(%d,%d) error=%.06f, delta=%.09f",
                         n, param_idx, sign_idx, s_new[x_opt], s_delta)

            x = x_save.copy()
            n += 1
            s = s_new[x_opt]
        end = time()
        elapsed = end-start

        if n >= self.maxit:
            reason = "Maximum iterations exceeded."
        elif s_delta < self.tolerance:
            reason = "Error reduction below tolerance."
        elif step_size < self.step_min:
            reason = "Step size smaller than minimum."
        else:
            reason = "Unknown. Termination conditions not met."

        # save final parameters back to model
        log.info("Coord. Descent finished:\n"
                 "Reason: {0}\n"
                 "Step size: {1:.09f}.\n"
                 "Steps: {2}.\n"
                 "Time elapsed: {3} seconds."
                 .format(reason, step_size, n, elapsed))
        #log.debug("Optimized parameters: %s", str(x))
        phi = vector_to_phi(x, self.phi0)
        self.stack.set_phi(phi)
        log.info("Final MSE: {0}".format(s))

        return(s)


class fit_iteratively(nems_fitter):
    """
    iterate through modules, running fitting each one with sub_fitter()

    TODO: update class name to FitIteratively per pep8 guidelines.

    """

    name = 'fit_iteratively'
    sub_fitter = None
    max_iter = 100
    module_sets=[]
    tolerance=0.000001

    def my_init(self, sub_fitter=basic_min, max_iter=100,
                min_kwargs={'routine': 'L-BFGS-B', 'maxit': 10000}):
        self.sub_fitter = sub_fitter(self.stack, **min_kwargs)
        self.max_iter = max_iter
        self.module_sets = [[i] for i in self.fit_modules]

    def do_fit(self):
        self.sub_fitter.tolerance = self.tolerance
        itr = 0
        err = self.stack.error()
        this_itr = 0

        while itr < self.max_iter:
            this_itr += 1

            for i in self.module_sets:
                log.info("Begin sub_fitter on mod: {0}; iter {1}; tol={2}"
                         .format(self.stack.modules[i[0]].name, itr,
                                 self.sub_fitter.tolerance))
                self.sub_fitter.fit_modules = i
                new_err = self.sub_fitter.do_fit()
            if err - new_err < self.sub_fitter.tolerance:
                log.info("\nError improvement less than tol,"
                         "starting new outer iteration")
                itr += 1
                self.sub_fitter.tolerance = self.sub_fitter.tolerance / 2
                this_itr = 0
            elif this_itr > 20:
                log.info("\nToo many loops at this tolerance, stuck?")
                itr += 1
                self.sub_fitter.tolerance = self.sub_fitter.tolerance / 2
                this_itr = 0

            err = new_err

        # Fit all params together aferward. If iterative fit did its job,
        # this should be a very short operation.
        # May only be useful for testing.
        log.debug("Subfitting complete, beginning whole-model fit...")
        self.sub_fitter.fit_modules = self.fit_modules
        err = self.sub_fitter.do_fit()

        # These should match
        log.debug("self.stack.error() is: {0}\n"
                  "local err variable is: {1}\n"
                  .format(self.stack.error(), err))

        #return(self.stack.error())
        return err


class fit_by_type(nems_fitter):
    """
    Iterate through modules, fitting each module with a different sub fitter
    that depends on the type of each module, i.e. if it is a nonlinearity,
    fir filter, etc...

    min_kwargs should be a dictionary of dictionaries:
        min_kwargs={'basic_min':{'routine':'L-BFGS','maxit':10000},
                    'anneal_min':{'min_method':'L-BFGS-B','anneal_iter':100,
                                  'stop':5,'maxiter':10000,'up_int':10,
                                  'bounds':None, 'temp':0.01,'stepsize':0.01},
                    etc...}
    Note that all of these fields need not be filled out, but if this is the
    case the subfitters will use their default settings.

    """

    name = 'fit_by_type'
    maxiter = 5
    tolerance=0.000001
    fir_filter_sfit = None
    nonlinearity_sfit = None
    weight_channels_sfit = None
    state_gain_sfit = None

    # NOTES: Possible improvements?
    #        Could start with a rough fit on each module with different sub
    #           fitters to find the one that performs best, instead of
    #           hardcoding the subfitters for each module.
    #           (Can still leave in the option to override).
    #        Ex: pass in a dict of sub fitters and their args
    #           for each module, fit only those module params (like iter fit)
    #           using each sub_fitter specified and a small number of
    #           iterations. keep the one with the best
    #           score for that module.
    #           After optimal sub fitter found for each module, do full
    #           iterative fit.
    #        Pros: removes need to establish relationship between
    #           module type and fitter, and always finds th best sub fitter
    #           instead of relying on user to know which one is best.
    #        Cons: Fit will take a bit longer since it has to do several
    #           mini fits before doing the existing routine. However, this
    #           shouldn't add *that* much time as long as the smaller iter
    #           count is reasonable.

    # Implementing above changes as BestMatch fitter class. -jacob 1/3/2018

    def my_init(self, fir_filter_sfit=basic_min, nonlinearity_sfit=anneal_min, weight_channels_sfit=basic_min,
                state_gain_sfit=basic_min, maxiter=5, min_kwargs={'basic_min': {'routine': 'L-BFGS-B', 'maxit': 10000}, 'anneal_min':
                                                                  {'min_method': 'L-BFGS-B', 'anneal_iter': 100, 'stop': 5, 'maxiter': 10000, 'up_int': 10, 'bounds': None,
                                                                   'temp': 0.01, 'stepsize': 0.01, 'verb': False}}):
        self.fir_filter_sfit = fir_filter_sfit(
            self.stack, **min_kwargs[fir_filter_sfit.name])
        self.nonlinearity_sfit = nonlinearity_sfit(
            self.stack, **min_kwargs[nonlinearity_sfit.name])
        self.weight_channels_sfit = weight_channels_sfit(
            self.stack, **min_kwargs[weight_channels_sfit.name])
        self.state_gain_sfit = state_gain_sfit(
            self.stack, **min_kwargs[state_gain_sfit.name])
        self.maxiter = maxiter

        self.modname_to_fitter = {
                'filters.fir': self.fir_filter_sfit,
                'nonlin.gain': self.nonlinearity_sfit,
                'filters.weight_channels': self.weight_channels_sfit,
                'pupil.pupgain': self.state_gain_sfit,
                }

    def do_fit(self):
        itr = 0
        err = self.stack.error()
        while itr < self.maxiter:
            self.fir_filter_sfit.tolerance = self.tolerance
            self.nonlinearity_sfit.tolerance = self.tolerance
            self.weight_channels_sfit.tolerance = self.tolerance
            self.state_gain_sfit.tolerance = self.tolerance

            # More or less the same as iterative fit except for
            # changing the sub_fitter.
            for i in self.fit_modules:
                name = self.stack.modules[i].name
                try:
                    sub_fitter = self.modname_to_fitter[name]
                except:
                    sub_fitter = self.fir_filter_sfit
                    log.info("Couldn't find a sub fitter for %s module,"
                             "using %s instead.",
                             name, sub_fitter.name
                             )

                log.info('Sub-fitting on %s module with %s'
                         .format(name, sub_fitter.name))
                log.info('Current iter: %d', itr)
                log.info('Current tolerance: %.09f', self.tolerance)
                sub_fitter.fit_modules = [i]
                new_err = sub_fitter.do_fit()

            if err - new_err < self.tolerance:
                log.info("\nError improvement less than tolerance,"
                         "starting new outer iteration")
                itr += 1
                self.tolerance = self.tolerance/2

            err = new_err

        log.info("Fit by type finished, final error: {0}"
                 .format(err))
        return(err)

class BestMatch(nems_fitter):
    """Iterate through modules, performing a rough fit on each module
    using each of the sub fitters specified. Then, iterate once more
    doing a complete fit for each module using the sub fitter that
    performed best on the rough fit.

    sub_fitters should be a dictionary of dictionaries:
        sub_fitters={nems.fitters.basic_min:{'routine':'L-BFGS','maxit':10000},
                    nems.fitters.anneal_min:{'min_method':'L-BFGS-B',
                                             'anneal_iter':100, 'stop':5,
                                             'maxiter':10000,'up_int':10,
                                             'bounds':None, 'temp':0.01,
                                             'stepsize':0.01},
                    nems.fitters.CoordinateDescent: {}, #use defaults
                    etc...}

    """

    name = 'BestMatch'
    # set at 1 for now for testing to help speed.
    # but maxiter may not even make sense for this one, since
    # the full fit should always be done after 1 iteration. If more
    # iters are desired, can just increase the maxit args passed to each
    # sub fitter.
    maxiter = 1
    tolerance = 0.000001
    rough_tolerance = 0.001 # Smaller values here may run very slow
    sub_fitters={
        basic_min: {'routine': 'L-BFGS-B', 'maxit': 10000},
        anneal_min: {
            'min_method': 'L-BFGS-B','anneal_iter': 100, 'stop': 5,
            'maxiter': 10000, 'up_int': 10, 'bounds': None,
            'temp': 0.01, 'stepsize': 0.01, 'verb': False
            }
        }

    def my_init(self, maxiter=1,
                sub_fitters={
                    basic_min: {'routine': 'L-BFGS-B', 'maxit': 10000},
                    anneal_min: {
                        'min_method': 'L-BFGS-B','anneal_iter': 100, 'stop': 5,
                        'maxiter': 10000, 'up_int': 10, 'bounds': None,
                        'temp': 0.01, 'stepsize': 0.01, 'verb': False
                        },
                    CoordinateDescent: {}, #use defaults
                    }):

        self.maxiter = maxiter
        self.phi0 = self.stack.get_phi()
        self.sub_fitters = sub_fitters
        self.sub_fitter_list = []
        self.best_fitters = {}

    def _set_sub_fitters(self, new_tol):
        """Convert fitter classes (keys) and kwargs (values) from
        sub_fitters dict to list of fitter instances, and set new
        tolerance for each fitter.

        """

        self.sub_fitter_list = [
                fitter(self.stack, **kwargs)
                for fitter, kwargs in self.sub_fitters.items()
                ]
        for fitter in self.sub_fitter_list:
            setattr(fitter, 'tolerance', new_tol)

    def do_fit(self):
        # Find the best fitter for each module
        for i in self.fit_modules:
            mod = self.stack.modules[i]
            scores = []
            # Reset fitter tolerance and instances in between iterations
            # incase fitter routines change settings.
            self._set_sub_fitters(self.rough_tolerance)

            for fitter in self.sub_fitter_list:
                # Reset stack phi in between fitters. Otherwise,
                # better fits might get stuck in local mins found by
                # worse fits.
                self.stack.set_phi(self.phi0)
                log.info('Performing rough fit on %s module with %s',
                         mod.name, fitter.name)
                log.info('Current tolerance: %.09f', self.rough_tolerance)
                fitter.fit_modules = [i]
                score = fitter.do_fit()
                log.info('Score for %s from %s was: %.09f',
                         mod.name, fitter.name, score)
                scores.append(score)

            best_idx = scores.index(min(scores))
            best_fitter = self.sub_fitter_list[best_idx]
            log.info('Best fitter for %s was: %s, with a score of: %.09f',
                     mod.name, best_fitter.name, min(scores))
            self.best_fitters[mod.name] = best_idx

        # Perform the full fit using the best fitters identified
        # in previous loop.
        self.stack.set_phi(self.phi0)
        self._set_sub_fitters(self.tolerance)
        err = self.stack.error()
        itr = 0

        log.info("Beginning full fit, maxiter: %d", self.maxiter)
        while itr < self.maxiter:
            for i in self.fit_modules:
                # Grab best fitter
                mod = self.stack.modules[i]
                fitter_idx = self.best_fitters[mod.name]
                fitter = self.sub_fitter_list[fitter_idx]
                log.info("Performing full fit on %s with %s",
                         mod.name, fitter.name)
                new_err = fitter.do_fit()
                delta = err - new_err
                log.info("Score for %s from %s was: %.09f.\n"
                         "Improvement over last score: %.09f.",
                         mod.name, fitter.name, new_err, delta
                         )

                if delta < 0:
                    log.debug("Something went wrong,"
                              "error got worse: %.09f to %.09f",
                              err, new_err)
                elif delta < self.tolerance:
                    log.info("\nError improvement less than tolerance,"
                             "starting new outer iteration")
                    self.tolerance *= 0.5
                    break

                err = new_err
            itr += 1

        log.info("BestMatch fit finished, final error: {0}"
                 .format(err))
        for i in self.fit_modules:
            mod = self.stack.modules[i]
            fit_idx = self.best_fitters[mod.name]
            fitter = self.sub_fitter_list[fit_idx]
            log.debug("Fitter used for %s was: %s",
                      mod.name, fitter.name)
        return(err)


class SequentialFit(nems_fitter):
    """Fits each parameter in order, one at a time -- similar to iterative fit,
    but per parameter instead of per module. Also similar to
    coordinate descent, but only looks at one parameter instead of all.

    """

    name = 'SequentialFit'
    maxit = 100
    tolerance = 0.00000001
    step_init = 0.1 # values will change by step*value
    step_change = 0.5
    step_min = 1e-7

    def my_init(self, tolerance=0.000001, maxit=100):
        self.maxit = maxit
        self.tolerance = tolerance

    def cost_fn(self, vector):
        phi = vector_to_phi(vector, self.phi0)
        self.stack.set_phi(phi)
        self.stack.evaluate(self.fit_modules[0])
        mse = self.stack.error()
        return(mse)

    def do_fit(self):
        self.phi0 = self.stack.get_phi()
        x0 = phi_to_vector(self.phi0)
        err = self.stack.error()
        x_saved = x0.copy()
        itr = 0

        log.info("Entering iterative loop, maxit: %d", self.maxit)
        while itr < self.maxit:
            pre_err = err
            log.info("Beginning iteration # %d", itr)
            for i, p in enumerate(x0):
                itr2 = 0
                direction = 1 # mult step by 1 or -1
                step = self.step_init
                while True:
                    #log.debug("Outer loop # %d", itr)
                    #log.debug("Inner loop # %d for pameter # %d,\n"
                    #          "Step size is: %s.\n"
                    #          "Direction is: %s.\n"
                    #          "Error is: %s.\n",
                    #          itr2, i, step, direction, err)

                    # step param in pos or neg direction
                    param = x_saved[i]
                    param_new = step*direction*param
                    x_new = x_saved.copy()
                    x_new[i] = param_new
                    new_err = self.cost_fn(x_new)
                    delta = err - new_err
                    #log.debug("Delta was: %.09f", delta)
                    itr2 += 1

                    # If error got worse or didn't change, reduce step
                    # size and change direction. Continue until min step.
                    if delta <= 0:
                        if step <= self.step_min:
                            log.debug("Error got worse and step is smaller"
                                     "than minimum, moving to next parameter.")
                            break
                        else:
                            direction *= -1
                            step *= self.step_change
                    # Otherwise, keep going in same direction
                    else:
                        x_saved[i] = param_new
                        err = new_err

            itr += 1
            post_err = err
            if post_err == pre_err:
                # If errors were the same, then no change to any parameter
                # resulted in a better score, so we're finished.
                log.info("Outer loop # %d, no change in error. Stopping fit.",
                         itr)
                break

        log.info("Fit finished, final error: %.09f", err)
        return err

# Notes for future changes:
#   -Generic function for reporting fit options might be useful, i.e.
#        tolerance, max iterations, etc.
#        Currently have to access fitter object attributes directly, which
#        might not always be named the same and not all fitters have the
#        the same relevant attributes.
#        ex:
#          self.reported_attrs = ['tolerance', 'maxit', 'my_attr', ...]
#          def report_attributes(self):
#              report = [getattr(self, a) for a in self.reported_attrs
#                        if hasattr(self, a)]
#              # do fitter specific stuff to gather extra info if needed
#              # and append info to report
#              # (maybe not everything is appropriate to define as an attr)
#              return report
#        mostly useful for testing/comparing, but would make it easier for
#        end users to look at fitter settings without having to scroll
#        through a bunch of code.
#        (Stuff like tolerance that gets defined in the base class can still
#        be pulled directly I guess, but most fitters willl likely define
#        new things too).
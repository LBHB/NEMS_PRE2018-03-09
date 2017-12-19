#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import os
import scipy as sp
import numpy as np

def phi_to_vector(phi):
    '''
    Convert a list of dictionaries where the values are scalars or array-like to
    a single vector.

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
            print('Eval #{0}. MSE={1}'.format(
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
    and outputs the current mean square error (mse). The cost function is evaulated
    by the scipy optimize.minimize routine, which seeks to minimize the mse by
    changing the function parameters.

    Scipy optimize.minimize is set to use the minimization algorithm 'L-BFGS-B'
    as a default, as this is memory efficient for large parameter counts and seems
    to produce good results. However, there are many other potential algorithms
    detailed in the documentation for optimize.minimize

    """

    name = 'basic_min'
    maxit = 50000
    routine = 'L-BFGS-B'

    def my_init(self, routine='L-BFGS-B', maxit=50000, tolerance=1e-7):
        """
        Initializes the fitter.

        routine: the algorithm that scipy.optimize.minimize should use. L-BFGS-B
                and SLSQP tend to work very well, while Nelder-Mead, Powell, and
                BFGS work, but not as well. The documentation for
                scipy.optimize.minimize has more details on algorithms that can
                be used

        maxit: maximum number of iterations for the fitter to use. Different
                than number of function evaluations
        tolerance: the "accuracy" to which the cost function is fit. E.g., if we
                have a tolerance of 0.001, the fitter will fit until the value of
                the cost function is stable at the third decimal place
        """
        print("initializing basic_min")
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
        if self.counter % 1000 == 0:
            print('Eval #' + str(self.counter))
            print('Error=' + str(err))
            self.tick_queue()  # This just updates the prgress indicator
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
        cons = ()

        # Below here are the general need for a nems_fitter object.
        self.phi0 = self.stack.get_phi()
        self.counter = 0
        vector = phi_to_vector(self.phi0)
        print("basic_min: phi0 initialized (fitting {0} parameters)" \
              .format(len(vector)))
        sp.optimize.minimize(self.cost_fn, vector, method=self.routine,
                             constraints=cons, options=opt, tol=self.tolerance)
        print("Final {0}: {1}".format(
            self.stack.modules[-1].name, self.stack.error()))
        print('           ')
        return(self.stack.error())


class anneal_min(nems_fitter):
    """
    A simulated annealing method to find the ~global~ minimum of your parameters.

    This fitter uses scipy.optimize.basinhopping, which is scipy's built-in annealing
    routine. Essentially, this routine uses scipy.optimize.minimize to minimize the
    function, then randomly perturbs the function and reminimizes it. It will continue
    this procedure until either the maximum number of iterations had been exceed or
    the minimum remains constant for a specified number of iterations.

    anneal_iter=number of annealing iterations to perform
    stop=number of iterations after which to stop annealing if global min remains the same
    up_int=update step size every up_int iterations
    maxiter=maximum iterations for each round of minimization
    tolerance=tolerance for each round of minimization
    min_method=method used for each round of minimization. 'L-BFGS-B' works well
    bounds should be [(xmin,xmax),(ymin,ymax),(zmin,zmax),etc]

    WARNING: this fitter takes a ~~long~~ time. It is usually better to try basic_min
    first, and then use this method if basic_min fails.

    Also, note that since basinhopping (at least as implemented here) uses random jumps,
    the results my not be exactly the same every time, and the annealing may take a
    different number of iterations each time it is called
    @author: shofer, 30 June 2017

    Further note: this is currently set up to take small jumps, as might be useful
    for fitting FIR filters or small nonlinearities. To use this fitter effectively,
    the "expected" value of the coefficients must be taken into account.
    --njs, 5 July 2017
    """

    name = 'anneal_min'
    anneal_iter = 100
    stop = 5
    up_int = 10
    min_method = 'L-BFGS-B'
    maxiter = 10000
    tolerance = 0.01

    def my_init(self, min_method='L-BFGS-B', anneal_iter=100, stop=5, maxiter=10000, up_int=10, bounds=None,
                temp=0.01, stepsize=0.01, verb=False):
        print("initializing anneal_min")
        self.anneal_iter = anneal_iter
        self.min_method = min_method
        self.stop = stop
        self.maxiter = 10000
        self.bounds = bounds
        self.up_int = up_int
        self.temp = temp
        self.step = stepsize
        self.verb = verb

    def cost_fn(self, phi):
        phi = vector_to_phi(vector, self.phi0)
        stack.set_phi(phi)
        self.stack.evaluate(self.fit_modules[0])
        err = self.stack.error()
        self.counter += 1
        if self.counter % 1000 == 0:
            print('Eval #' + str(self.counter))
            print('Error=' + str(err))
        return(err)

    def do_fit(self):
        opt = dict.fromkeys(['maxiter'])
        opt['maxiter'] = int(self.maxiter)
        opt['eps'] = 1e-7
        min_kwargs = dict(method=self.min_method,
                          tolerance=self.tolerance, bounds=self.bounds, options=opt)
        self.phi0 = self.stack.get_phi()
        self.counter = 0
        print("anneal_min: phi0 intialized (fitting {0} parameters)".format(
            len(self.phi0)))
        #print("maxiter: {0}".format(opt['maxiter']))
        opt_res = sp.optimize.basinhopping(self.cost_fn, self.phi0, niter=self.anneal_iter,
                                           T=self.temp, stepsize=self.step, minimizer_kwargs=min_kwargs,
                                           interval=self.up_int, disp=self.verb, niter_success=self.stop)
        phi_final = opt_res.lowest_optimization_result.x
        self.cost_fn(phi_final)
        print("Final MSE: {0}".format(self.stack.error()))
        print('           ')
        return(self.stack.error())


class coordinate_descent(nems_fitter):
    """
    coordinate descent - step one parameter at a time
    """

    name = 'coordinate_descent'
    maxit = 1000
    tolerance = 0.001
    step_init = 0.01
    step_change = 0.5
    step_min = 1e-7
    verbose = True

    def my_init(self, tolerance=0.001, maxit=1000, verbose=True):
        print("initializing basic_min")
        self.maxit = maxit
        self.tolerance = tolerance
        self.verbose = verbose

    def cost_fn(self, phi):
        phi = vector_to_phi(vector, self.phi0)
        stack.set_phi(phi)
        self.stack.evaluate(self.fit_modules[0])
        mse = self.stack.error()
        self.counter += 1
        # if self.counter % 100==0:
        #    print('Eval #{0}: Error={1}'.format(self.counter,mse))
        return(mse)

    def do_fit(self):
        raise NotImplementedError
        self.phi0 = self.stack.get_phi()
        self.counter = 0
        n_params = len(self.phi0)

#        if ~options.Elitism
#            options.EliteParams = n_params;
#            options.EliteSteps = 1;
#        end

        n = 1   # step counter
        x = self.phi0.copy()  # current phi
        x_save = x.copy()     # last updated phi
        s = self.cost_fn(x)   # current score
        # Improvement of score over the previous step
        s_new = np.zeros([n_params, 2])
        s_delta = np.inf     # Improvement of score over the previous step
        step_size = self.step_init  # Starting step size.
        #print("{0}: phi0 intialized (start error={1}, {2} parameters)".format(self.name,s,len(self.phi0)))
        # print(x)
        print("starting CD: step size: {0:.6f} tolerance: {1:.6f}".format(
            step_size, self.tolerance))
        while (s_delta < 0 or s_delta >
               self.tolerance) and n < self.maxit and step_size > self.step_min:
            for ii in range(0, n_params):
                for ss in [0, 1]:
                    x[:] = x_save[:]
                    if ss == 0:
                        x[ii] += step_size
                    else:
                        x[ii] -= step_size
                    s_new[ii, ss] = self.cost_fn(x)

            x_opt = np.unravel_index(s_new.argmin(), (n_params, 2))
            popt, sopt = x_opt
            s_delta = s - s_new[x_opt]

            if s_delta < 0:
                step_size = step_size * self.step_change
                # if self.verbose is True:
                print("{0}: Backwards (delta={1}), adjusting step size to {2}".format(
                    n, s_delta, step_size))

            elif s_delta < self.tolerance:
                if self.verbose is True:
                    print("{0}: Error improvement too small (delta={1}). Iteration complete.".format(
                        n, s_delta))

            elif sopt:
                x_save[popt] -= step_size
                if self.verbose is True:
                    print("{0}: best step={1},{2} error={3}, delta={4}".format(
                        n, popt, sopt, s_new[x_opt], s_delta))
            else:
                x_save[popt] += step_size
                if self.verbose is True:
                    print("{0}: best step={1},{2} error={3}, delta={4}".format(
                        n, popt, sopt, s_new[x_opt], s_delta))

            x = x_save.copy()
            n += 1
            s = s_new[x_opt]

        # save final parameters back to model
        print("done CD: step size: {0:.6f} steps: {1}".format(step_size, n))
        phi = vector_to_phi(x, self.phi0)
        stack.set_phi(phi)

        #print("Final MSE: {0}".format(s))
        return(s)


class fit_iteratively(nems_fitter):
    """
    iterate through modules, running fitting each one with sub_fitter()
    """

    name = 'fit_iteratively'
    sub_fitter = None
    max_iter = 5
    module_sets=[]

    def my_init(self, sub_fitter=basic_min, max_iter=5, min_kwargs={
                'routine': 'L-BFGS-B', 'maxit': 10000}):
        self.sub_fitter = sub_fitter(self.stack, **min_kwargs)
        self.max_iter = max_iter
        self.module_sets=[]
        for i in self.fit_modules:
            self.module_sets=self.module_sets + [i]

    def do_fit(self):
        self.sub_fitter.tolerance = self.tolerance
        itr = 0
        err = self.stack.error()
        this_itr = 0
        while itr < self.max_iter:
            this_itr += 1
            for i in self.module_sets:
                print("Begin sub_fitter on mod: {0}; iter {1}; tol={2}".format(self.stack.modules[i[0]].name,itr,self.sub_fitter.tolerance))
                self.sub_fitter.fit_modules = i
                new_err = self.sub_fitter.do_fit()
            if err - new_err < self.sub_fitter.tolerance:
                print("")
                print("error improvement less than tol, starting new outer iteration")
                itr += 1
                self.sub_fitter.tolerance = self.sub_fitter.tolerance / 2
                this_itr = 0
            elif this_itr > 20:
                print("")
                print("too many loops at this tolerance, stuck?")
                itr += 1
                self.sub_fitter.tolerance = self.sub_fitter.tolerance / 2
                this_itr = 0

            err = new_err

        return(self.stack.error())


class fit_by_type(nems_fitter):
    """
    Iterate through modules, fitting each module with a different sub fitter
    that depends on the type of each module, i.e. if it is a nonlinearity, fir filter,
    etc...

    min_kwargs should be a dictionary of dictionaries:
        min_kwargs={'basic_min':{'routine':'L-BFGS','maxit':10000},'anneal_min':
            {'min_method':'L-BFGS-B','anneal_iter':100,'stop':5,'maxiter':10000,'up_int':10,'bounds':None,
                'temp':0.01,'stepsize':0.01}, etc...}
    Note that all of these fields need not be filled out, but if this is the case the
    subfitters will use their default settings.
    """

    name = 'fit_by_type'
    maxiter = 5
    fir_filter_sfit = None
    nonlinearity_sfit = None
    weight_channels_sfit = None
    state_gain_sfit = None

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

    def do_fit(self):
        itr = 0
        err = self.stack.error()
        while itr < self.maxiter:
            self.fir_filter_sfit.tolerance = self.tolerance
            self.nonlinearity_sfit.tolerance = self.tolerance
            self.weight_channels_sfit.tolerance = self.tolerance
            self.state_gain_sfit.tolerance = self.tolerance
            for i in self.fit_modules:
                name = self.stack.modules[i].name
                print('Sub-fitting on {0} module with {1}'.format(name,
                                                                  getattr(getattr(self, name + '_sfit'), 'name')))
                print('Current iter: {0}'.format(itr))
                print('Current tolerance: {0}'.format(self.tolerance))
                setattr(getattr(self, name + '_sfit'), 'fit_modules', [i])
                new_err = getattr(self, name + '_sfit').do_fit()
            if err - new_err < self.tolerance:
                print("")
                print(
                    "error improvement less than tolerance, starting new outer iteration")
                itr += 1
                self.tolerance = self.tolerance / 2
            err = new_err
        return(self.stack.error())

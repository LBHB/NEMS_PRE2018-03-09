#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fitter keywords

Created on Fri Aug 11 11:14:52 2017

@author: shofer
"""

import logging
log = logging.getLogger(__name__)

import nems.modules as nm
import nems.fitters

from .registry import keyword_registry


def fit00(stack):
    """
    Fits the model parameters using a mean squared error loss function with
    the L-BFGS-B algorithm, to a cost function tolerance of 0.001.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error)
    # set error (for minimization) for this stack to be output of last module
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.basic_min(stack)
    stack.fitter.tolerance = 0.001
    stack.fitter.do_fit()


def fit01(stack):
    """
    Fits the model parameters using a mean squared error loss function with
    the L-BFGS-B algorithm, to a cost function tolerance of 10^-8.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.basic_min(stack)
    stack.fitter.tolerance = 0.00000001
    stack.fitter.do_fit()


def fit02(stack):
    """
    Fits the model parameters using a mean squared error loss function with
    the SLSQP algorithm, to a cost function tolerance of 10^-6.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.basic_min(stack, routine='SLSQP')
    stack.fitter.tolerance = 0.000001
    stack.fitter.do_fit()


def fit03(stack):
    """
    Fits the model parameters using a srinkage-mean squared error loss function with
    the L-BFGS-B algorithm, to a cost function tolerance of 10^-9.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error, shrink=0.1)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.basic_min(stack)
    stack.fitter.tolerance = 0.000000001
    stack.fitter.do_fit()


def fit04(stack):
    """
    Fits the model parameters using a Poisson LL loss function with
    the L-BFGS-B algorithm, to a cost function tolerance of 10^-7.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.likelihood_poisson)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.basic_min(stack)
    stack.fitter.tolerance = 0.0000001
    stack.fitter.do_fit()


def fit00h1(stack):
    """
    Fits the model parameters using a pseudo-huber loss function with
    the L-BFGS-B algorithm, to a cost function tolerance of 0.001.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.pseudo_huber_error, b=1.0)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.basic_min(stack)
    stack.fitter.tol = 0.001
    stack.fitter.do_fit()
    stack.popmodule()
    stack.append(nm.metrics.mean_square_error)


def fitannl00(stack):
    """
    Fits the model parameters using a simulated annealing fitting procedure.

    Each repetition of annealing is performed with a mean_square_error cost function
    using the L-BFGS-B algorithm, to a tolerance of 0.001.

    50 rounds of annealing are performed, with the step size updated dynamically
    every 10 rounds. The routine will stop if the minimum function value remains
    the same for 5 rounds of annealing.

    Note that this routine takes a long time.
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.anneal_min(
        stack, anneal_iter=50, stop=5, up_int=10, bounds=None)
    stack.fitter.tol = 0.001
    stack.fitter.do_fit()


def fitannl01(stack):
    """
    Fits the model parameters using a simulated annealing fitting procedure.

    Each repetition of annealing is performed with a mean_square_error cost function
    using the L-BFGS-B algorithm, to a tolerance of 10^-6.

    100 rounds of annealing are performed, with the step size updated dynamically
    every 5 rounds. The routine will stop if the minimum function value remains
    the same for 10 rounds of annealing.

    Note that this routine takes a very long time.
    """
    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.anneal_min(
        stack, anneal_iter=100, stop=10, up_int=5, bounds=None)
    stack.fitter.tol = 0.000001
    stack.fitter.do_fit()


def fititer00(stack):
    """
    Fits the model parameters using a mean-squared-error loss function with
    a coordinate descent algorithm. However, rather than fitting all model
    parameters at once, it only fits the parameters for one model at a time.
    The routine fits each module to a tolerance of 0.001, than halves the tolerance
    and repeats up to 9 more times.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error, shrink=0.5)
    stack.error = stack.modules[-1].error

    stack.fitter = nems.fitters.fitters.fit_iteratively(stack, max_iter=5)
    # stack.fitter.sub_fitter=nems.fitters.fitters.basic_min(stack)
    stack.fitter.sub_fitter = nems.fitters.fitters.coordinate_descent(
        stack, tolerance=0.001, maxit=10, verbose=False)
    stack.fitter.sub_fitter.step_init = 0.05

    stack.fitter.do_fit()


def fititer01(stack):
    """
    Fits the model parameters using a mean-squared-error loss function with
    a coordinate descent algorithm. However, rather than fitting all model
    parameters at once, it only fits the parameters for one model at a time.
    The routine fits each module to a tolerance of 0.001, than halves the tolerance
    and repeats up to 9 more times.

    Should be appended last in a modelname (excepting "nested" keywords)
    """
    stack.append(nm.metrics.mean_square_error, shrink=0.5)
    stack.error = stack.modules[-1].error

    stack.fitter = nems.fitters.fitters.fit_iteratively(stack, max_iter=5)
    stack.fitter.sub_fitter = nems.fitters.fitters.basic_min(stack)

    stack.fitter.do_fit()

def fitcoord00(stack):
    """Fits model parameters using greedy coordinate desecent algorithm.
    Cost function uses mean squared error, fitter settings left as defaults.

    Note: recommended to use this kw for testing Coordinate Descent, until
    annealing and pseudo-caching are either fixed or detrimented.

    """

    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error

    stack.fitter = nems.fitters.fitters.coordinate_descent(stack)
    stack.fitter.do_fit()

def fitcoord01(stack):
    """Fits model parameters using greedy coordinate desecent algorithm.
    Cost function uses mean squared error.
    Also enables annealing with anneal count = 30.

    Note: Annealing 'works', but so far hasn't helped performance.

    """

    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error

    stack.fitter = nems.fitters.fitters.coordinate_descent(
            stack, anneal=30
            )
    stack.fitter.do_fit()

def fitcoord02(stack):
    """Fits model parameters using greedy coordinate desecent algorithm.
    Cost funciton uses mean squared error.
    Also enables pseudo-caching of prior module evals.

    Note: Pseudo-caching not yet matching non-cached results.

    """

    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error

    stack.fitter = nems.fitters.fitters.coordinate_descent(
            stack, pseudo_cache=True,
            )
    stack.fitter.do_fit()

def fitcoord03(stack):
    """Fits model parameters using greedy coordinate desecent algorithm.
    Cost function uses mean squared error.
    Also enables pseudo-caching of prior module evals, and
    enables annealing with anneal count = 30.

    Note: Annealing 'works', but so far hasn't helped performance.
          Pseudo-caching not yet matching non-cached results.

    """

    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error

    stack.fitter = nems.fitters.fitters.coordinate_descent(
            stack, pseudo_cache=True, anneal=30,
            )
    stack.fitter.do_fit()

def skopt00(stack):
    """Fits model parameters using Scikit-Optimize's gp_minimize."""

    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.SkoptMin(stack)
    stack.fitter.tolerance = 0.000001
    stack.fitter.do_fit()

def skopt01(stack):
    """Fits model parameters using Scikit-Optimize's forest_minimize."""

    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.SkoptForestMin(stack)
    stack.fitter.tolerance = 0.000001
    stack.fitter.do_fit()

def skopt02(stack):
    """Fits model parameters using Scikit-Optimize's gbrt_minimize."""

    stack.append(nm.metrics.mean_square_error)
    stack.error = stack.modules[-1].error
    stack.evaluate(2)

    stack.fitter = nems.fitters.fitters.SkoptGbrtMin(stack)
    stack.fitter.tolerance = 0.000001
    stack.fitter.do_fit()


matches = ['fit', 'skopt']

for k, v in list(locals().items()):
    for m in matches:
        if k.startswith(m) and callable(v):
            keyword_registry[k] = v

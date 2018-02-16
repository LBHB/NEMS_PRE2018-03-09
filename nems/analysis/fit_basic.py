import logging
from functools import partial

from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize
import nems.priors
import nems.fitters.mappers
import nems.modelspec
import nems.metrics.api
import nems.segmentors

def fit_basic(data, modelspec,
              fitter=coordinate_descent,
              segmentor=nems.segmentors.use_all_data,
              mapper=nems.fitters.mappers.simple_vector,
              metric=lambda data: nems.metrics.api.mse(
                                {'pred': data.get_signal('pred').as_continuous(),
                                 'resp': data.get_signal('resp').as_continuous()}
                                )):
    '''
    Required Arguments:
     data          A recording object
     modelspec     A modelspec object

    Optional Arguments:
     fitter        A function of (sigma, costfn) that tests various points,
                   in fitspace (i.e. sigmas) using the cost function costfn,
                   and hopefully returns a better sigma after some time.
     mapper        A class that has two methods, pack and unpack, which define
                   the mapping between modelspecs and a fitter's fitspace.
     segmentor     An function that selects a subset of the data during the
                   fitting process. This is NOT the same as est/val data splits
     metric        A function of a Recording that returns an error value
                   that is to be minimized.

    Returns
     A list containing a single modelspec, wich has the best parameters found
    by this fitter.
    '''

    # Create the mapper object that translats to and from modelspecs.
    # It has two methods that, when defined as mathematical functions, are:
    #    .pack(modelspec) -> fitspace_point
    #    .unpack(fitspace_point) -> modelspec
    packer, unpacker = mapper(modelspec)

    # A function to evaluate the modelspec on the data
    evaluator = nems.modelspec.evaluate

    # TODO - unpacks sigma and updates modelspec, then evaluates modelspec
    #        on the estimation/fit data and
    #        uses metric to return some form of error
    def cost_function(sigma, unpacker, modelspec, data,
                      evaluator, metric):
        updated_spec = unpacker(sigma)
        # The segmentor takes a subset of the data for fitting each step
        # Intended use is for CV or random selection of chunks of the data
        data_subset = segmentor(data)
        updated_data_subset = evaluator(data_subset, updated_spec)
        error = metric(updated_data_subset)
        #print("inside cost function, current error: {}".format(error))
        #print("\ncurrent sigma: {}".format(sigma))
        return error

    # Freeze everything but sigma, since that's all the fitter should be
    # updating.
    cost_fn = partial(cost_function,
                      unpacker=unpacker, modelspec=modelspec,
                      data=data, evaluator=evaluator,
                      metric=metric)

    # get initial sigma value representing some point in the fit space
    sigma = packer(modelspec)

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn)
    improved_modelspec = unpacker(improved_sigma)
    results = [improved_modelspec]

    return results


def fit_random_subsets(data, modelspec, nsplits=1, rebuild_every=10000):
    '''
    Randomly picks a small fraction of the data to fit on.
    Intended to speed up initial converge on fitting large data sets.
    To improve efficiency, you may generally good to use the same subset
    for a bunch of cost function evaluations in a row.
    '''
    maker = nems.segmentors.random_jackknife_maker
    segmentor = maker(nsplits=nsplits, rebuild_every=rebuild_every,
                      invert=True, excise=True)
    return fit_basic(data, modelspec,
                     segmentor=segmentor)


def fit_jackknifes(data, modelspec, njacks=10):
    '''
    Takes njacks jackknifes, where each jackknife has some small
    fraction of data NaN'd out, and fits modelspec to them.
    '''
    models = []
    for i in range(njacks):
        logging.info("Fitting jackknife {}/{}".format(i, njacks))
        jk = data.jackknife_by_time(njacks, i)
        models += fit_basic(jk, modelspec, fitter=scipy_minimize)

    return models


def fit_subsets(data, modelspec, nsplits=10):
    '''
    Divides the data evenly into nsplits pieces, and fits a model
    to each of the pieces.
    '''
    models = []
    for i in range(nsplits):
        logging.info("Fitting subset {}/{}".format(i, nsplits))
        split = data.jackknife_by_time(nsplits, i, invert=True, excise=True)
        models += fit_basic(split, modelspec, fitter=scipy_minimize)

    return models


def fit_from_priors(data, modelspec, ntimes=10):
    '''
    Fit ntimes times, starting from random points sampled from the prior.
    '''
    models = []
    for i in range(ntimes):
        logging.info("Fitting from random start: {}/{}".format(i, ntimes))
        ms = nems.priors.set_random_phi(modelspec)
        models += fit_basic(data, ms, fitter=scipy_minimize)

    return models

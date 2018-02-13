from functools import partial

from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize
import nems.priors
import nems.fitters.mappers
import nems.modelspec
import nems.metrics.api

def fit_basic(data, modelspec,
              fitter=coordinate_descent,
              segmentor=lambda data: data,  # Default pass-thru
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

    # The segmentor takes a subset of the data for fitting on
    # Intended use is for CV or random selection of chunks of the data
    data_subset = segmentor(data)

    # A function to evaluate the modelspec on the data
    evaluator = nems.modelspec.evaluate

    # TODO - unpacks sigma and updates modelspec, then evaluates modelspec
    #        on the estimation/fit data and
    #        uses metric to return some form of error
    def cost_function(sigma, unpacker, modelspec, data_subset,
                      evaluator, metric):
        updated_spec = unpacker(sigma)
        updated_data_subset = evaluator(data_subset, updated_spec)
        error = metric(updated_data_subset)
        #print("inside cost function, current error: {}".format(error))
        #print("\ncurrent sigma: {}".format(sigma))
        return error

    # Freeze everything but sigma, since that's all the fitter should be
    # updating.
    cost_fn = partial(cost_function,
                      unpacker=unpacker, modelspec=modelspec,
                      data_subset=data_subset, evaluator=evaluator,
                      metric=metric)

    # get initial sigma value representing some point in the fit space
    sigma = packer(modelspec)

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn)
    improved_modelspec = unpacker(improved_sigma)
    results = [improved_modelspec]

    return results


# TODO: Remove this?
def fit_samples(data, modelspec, n_samples=1,
                fitter=coordinate_descent,
                segmentor=lambda data: data,  # Default pass-thru
                mapper=nems.fitters.mappers.simple_vector,
                metric=lambda data: nems.metrics.api.mse(
                                {'pred': data.get_signal('pred').as_continuous(),
                                 'resp': data.get_signal('resp').as_continuous()}
                                )):
    raise NotImplementedError

    for i in range(n_samples):
        # TODO: implement the sample_phi function in nems.priors
        this_mspec = nems.priors.set_random_phi(modelspec)
        this_data = data.copy()
        best_models = fit_basic(this_data, this_mspec, fitter, segmentor,
                                mapper, metric)
        pred = nems.modelspec.evaluate(this_data, best_models[0])
        err = metric(pred)
        

    return result


from functools import partial

from nems.fitters.api import dummy_fitter, coordinate_descent
import nems.fitters.mappers
import nems.modules.evaluators
import nems.metrics.api


def fit_basic(data, modelspec,
              fitter=dummy_fitter,
              segmentor=lambda data: data,
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
     fitter        TODO
     mapper        A class that has two methods, pack and unpack, which describe
                   the mapping between modelspecs and a fitter's fitspace.
     segmentor     An function that selects a subset of the data during the
                   fitting process.
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
    est_data = segmentor(data)

    # TODO - evaluates the data using the modelspec, then updates data['pred']
    evaluator = nems.modules.evaluators.matrix_eval

    # TODO - unpacks sigma and updates modelspec, then evaluates modelspec
    #        on the estimation/fit data and
    #        uses metric to return some form of error
    def cost_function(unpacker, modelspec, est_data, evaluator, metric,
                      sigma=None):
        updated_spec = unpacker(sigma)
        updated_est_data = evaluator(est_data, updated_spec)
        error = metric(updated_est_data)
        return error

    # Freeze everything but sigma, since that's all the fitter should be
    # updating.
    cost_fn = partial(
            cost_function, unpacker=unpacker, modelspec=modelspec,
            est_data=est_data, evaluator=evaluator, metric=metric,
            )

    # get initial sigma value representing some point in the fit space
    sigma = packer(modelspec)

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn)
    improved_modelspec = unpacker(improved_sigma)
    results = [improved_modelspec]

    return results

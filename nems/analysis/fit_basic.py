from functools import partial

from nems.fitters.api import dummy_fitter, bit_less_dummy_fitter, \
                             coordinate_descent
import nems.fitters.mappers
import nems.modules.evaluators
import nems.metrics.api

"""
# ----------------------------------------------------------------------------
# DEFINE THE COST FUNCTION
#
# Goal: Define the cost function and metric for use by the fitter.
# Please see "docs/architecture.svg" for more information.

# Option 1: Use mean squared error when fitting:
metric = lambda data: nems.metrics.MSE(data['resp'], data['pred'])

# Option 2: Use log-likelihood, if you predicted a gaussian at each point
# metric = lambda data: nems.metrics.LogLikelihood(data['resp'], data['pred'], data['pred_stddev'])

# Option 3: Use some other metric that you think is better
# metric = lambda data: nems.metrics.coherence(data['resp'], data['pred'])

# Finally, define the evaluator and cost functions
# TODO: I think these can be boilerplate elsewhere

# TODO: what's this meant to be? -jacob
from nems.model import generate_model

# If we're doing incremential fitting (for example)

class Phi:

    def __init__(self, phi):
        self.phi = phi
        self.free_parameters = None#

    def select_for_fit(self):
        fit_phi = []
        for module_phi, module_free_parameters in zip(self.phi, self.free_parameters):
            pass


for i, mod in enumerate(modelspec):
    eval_fn = model.compose_eval(modelspec[:i])
    phi = model.initialize_phi(modelspec)
    cost_fn = partial(nems.metrics.mse, eval_fn=eval_fn, pred_name='pred', resp_name='resp')

eval_fn = compose_transform(modelspec)

evaluator = generate_evaluation

evaluator = lambda data, mspec : nems.model.Model(mspec).evaluate(data, mspec)
cost_fn = lambda mspec: metric(evaluator(est, mspec))
"""

# Leaving above code for reference, redoing as function below to match
# signature in demo_script2.py and architecture.svg in planning  -jacob
def fit_basic(data, modelspec):
    # Data set (should be a recording object)
    # Modelspec: dict with the initial module specifications
    # Per architecture doc, analysis function should only take these two args

    # TODO: should this be exposed as an argument?
    # Specify how the data will be split up
    segmentor = lambda data: data.split_at_time(0.8)

    # TODO: should mapping be exposed as an argument?
    # get funcs for translating modelspec to and from fitter's fitspace
    # packer should generally take only modelspec as arg,
    # unpacker should take type returned by packer + modelspec
    packer, unpacker = nems.fitters.mappers.simple_vector()

    # split up the data using the specified segmentor
    est_data, val_data = segmentor(data)

    # bit hacky at the moment, but trying not to interfere with or rewrite mse
    # for now (which expects a dict of arrays) -jacob
    metric = lambda data: nems.metrics.api.mse(
                                {'pred': data.get_signal('pred').as_continuous(),
                                 'resp': data.get_signal('resp').as_continuous()}
                                )

    # TODO - evaluates the data using the modelspec, then updates data['pred']
    evaluator = nems.modules.evaluators.matrix_eval

    # TODO - unpacks sigma and updates modelspec, then evaluates modelspec
    #        on the estimation/fit data and
    #        uses metric to return some form of error
    def cost_function(unpacker, modelspec, est_data, evaluator, metric,
                      sigma=None):
        updated_spec = unpacker(sigma, modelspec)
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

    # TODO: should fitter be exposed as an argument?
    #       would make sense if exposing space mapper, since fitter and mapper
    #       type are related.
    fitter = coordinate_descent

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    improved_sigma = fitter(sigma, cost_fn)

    improved_modelspec = unpacker(improved_sigma, modelspec)
    results = [improved_modelspec]

    return results
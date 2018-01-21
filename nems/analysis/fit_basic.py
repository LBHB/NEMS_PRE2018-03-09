from functools import partial

import nems

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


# Leaving above code for reference, redoing as function below to match
# signature in demo_script2.py and architecture.svg in planning  -jacob
def fit_basic(data, modelspec):
    # Data set (should be a recording object)
    # Modelspec: dict with the initial module specifications
    # Per architecture doc, analysis function should only take these two args

    # TODO: should this be exposed as an optional argument?
    # Specify how the data will be split up
    segmentor = lambda data: data.split_at_time(0.8)

    # TODO: should mapping be exposed as an optional argument?
    # get funcs for translating modelspec to and from fitter's fitspace
    # packer should generally take only modelspec as arg,
    # unpacker should take modelspec + whatever type the packer returns
    packer, unpacker = nems.fitters.mappers.simple_vector()

    # split up the data using the specified segmentor
    est_data, val_data = segmentor(data)

    metric = lambda data: nems.metrics.MSE(data['resp'], data['pred'])
    # TODO - evaluates the data using the modelspec, then updates data['pred']
    evaluator = some_function_that_takes(data, modelspec)

    # TODO - unpacks sigma and updates modelspec, then evaluates modelspec and
    #        uses metric to return some form of error
    cost_fn = some_function_that_takes(sigma, unpacker, modelspec, evaluator, metric)

    # get initial sigma value representing some point in the fit space
    sigma = packer(modelspec)

    # TODO: should fitter be exposed as an optional argument?
    #       would make sense if exposing space mapper, since fitter and mapper
    #       type are related.
    fitter = partial(nems.fitters.gradient_descent(sigma, cost_fn,
                                           bounds=None, fixed=None))

    # Results should be a list of modelspecs
    # (might only be one in list, but still should be packaged as a list)
    results = fitter()

    return results
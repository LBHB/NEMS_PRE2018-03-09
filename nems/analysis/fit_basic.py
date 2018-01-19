
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

from nems.model import generate_model

# If we're doing incremential fitting (for example)

class Phi:

    def __init__(self, phi):
        self.phi = phi
        self.free_parameters = #

    def select_for_fit(self):
        fit_phi = []
        for module_phi, module_free_parameters in zip(self.phi, self.free_parameters):


for i in range(len(modelspec)):
    eval_fn = model.compose_eval(modelspec[:i])
    phi = model.initialize_phi(modelspec)
    cost_fn = partial(nems.metrics.mse, eval_fn=eval_fn, pred_name='pred', resp_name='resp')

eval_fn = compose_transform(modelspec)

evaluator = generate_evaluation

evaluator = lambda data, mspec : nems.model.Model(mspec).evaluate(data, mspec)
cost_fn = lambda mspec: metric(evaluator(est, mspec))

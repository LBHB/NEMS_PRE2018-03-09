from scipy import optimize

from .fitter import Fitter
from .util import phi_to_vector, vector_to_phi


class ScipyFitter(Fitter):

    def __init__(self, cost_function, phi_initialization='expected',
                 **optimize_kw):
        self.cost_function = cost_function
        self.phi_initialization = phi_initialization
        self.optimize_kw = optimize_kw

    def fit(self, model, signals):
        phi = self.get_initial_phi(model)
        vector = phi_to_vector(phi)
        self.phi_template = phi
        return optimize.fmin(self.cost_function, vector, args=(model, signals),
                             **self.optimize_kw)

    def get_initial_phi(self, model):
        priors = model.get_priors()
        phi = []
        for module_priors in priors:
            if self.phi_initialization == 'expected':
                module_phi = {k: v.mean() for k, v in module_priors}
            elif self.phi_initialization == 'random':
                module_phi = {k: v.sample() for k, v in module_priors}
            phi.append(module_phi)
        return phi

    def evalauate(self, vector, model, signals):
        phi = vector_to_phi(vector, self.phi_template)
        return self.cost_function(model, signals, phi)

class ScipyMinimizeFitter(ScipyFitter):
    """Fits the model using scipy.optimize.minimize instead of .fmin.
    TODO: Other ways to do this, but this seemed the simplest for now.
          -jacob, 1-13-18

    """

    def fit(self, model, signals):
        phi = self.get_initial_phi(model)
        vector = phi_to_vector(phi)
        self.phi_template = phi
        return optimize.minimize(self.cost_function, vector,
                                 args=(model, signals), **self.optimize_kw)

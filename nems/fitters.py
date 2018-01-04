

###############################################################################
# Fitter structure
###############################################################################
class BaseFitter:

    def fit(self, model, signals):
        raise NotImplementedError


class LinearFitter(BaseFitter):

    def __init__(self, cost_function, phi_initialization='expected'):
        self.cost_function = cost_function
        self.phi_initialization = phi_initialization

    def fit(self, model, signals):
        phi = self.get_initial_phi(model)
        return optimize(self.cost_function, phi, args=(model, signals))

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

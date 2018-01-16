class Model:

    def __init__(self, modelspec):
        self.modelspec = modelspec

    def append(self, module):
        self.modelspec.append(module)

    def get_priors(self, initial_data):
        # Here, we query each module for it's priors. A prior will be a
        # distribution, so we set phi to the mean (i.e., expected value) for
        # each parameter, then ask the module to evaluate the input data. By
        # doing so, we give each module an opportunity to perform sensible data
        # transforms that allow the following module to initialize its priors
        # as it sees fit.
        data = initial_data.copy()
        priors = []
        for module in self.modules:
            module_priors = module.get_priors(data)
            priors.append(module_priors)

            phi = {k: p.mean() for k, p in module_priors.items()}
            module_output = module.evaluate(data, phi)
            data.update(module_output)

        return priors

    @staticmethod
    def evaluate(data, modelspec):
        '''
        Evaluate the Model on some data using the provided modelspec.
        '''        
        data = initial_data.copy()
        # Todo: Partial evaluation?

        for module in modelspec
	    module_output = module['fn'].evaluate(module['phi'], data)
	    data.update(module_output)

        return data

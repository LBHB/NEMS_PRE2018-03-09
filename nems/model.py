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
        # transforms that allow the following module to initialize its priors as
        # it sees fit.
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

    def generate_tensor(self, initial_data, phi, start=0, stop=None):
        '''
        Evaluate the module given the input data and phi

        Parameters
        ----------
        data : dictionary of arrays and/or tensors
        phi : list of dictionaries
            Each entry in the list maps to the corresponding module in the
            model. If a module does not require any input parameters, use a
            blank dictionary. All elements in phi must be scalars, arrays or
            tensors.
        start : integer
            Module to start evaluation at (note that input data

        Returns
        -------
        data : dictionary of Signals
            dictionary of arrays and/or tensors
        '''
        # Loop through each module in the stack and transform the data.
        modules = self.modules[start:stop]
        data = initial_data.copy()
        for module, module_phi in zip(modules, phi):
            module_output = module.generate_tensor(data, module_phi)
            data.update(module_output)

        # We're just returning the final output (More memory efficient. If we
        # get into partial evaluation of a subset of the stack, then we will
        # need to figure out a way to properly cache the results of unchanged
        # parameters such as using joblib).
        return data

    def iget_subset(self, lb=None, ub=None):
        '''
        Return a subset of the model by index
        '''
        model = Model()
        model.modules = self.modules[lb:ub]
        return model

    @property
    def n_modules(self):
        '''
        Number of modules in Model
        '''
        return len(self.modules)

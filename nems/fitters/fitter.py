class Fitter:

    def fit(self, model, signals):
        raise NotImplementedError

def dummy_fitter(sigma, cost_fn, bounds=None, fixed=None):
    err = cost_fn(sigma)
    print("I did a 'fit'! err was: {0}".format(err))
    return sigma
import pymc3 as mc

from nems.distributions.api import Normal, HalfNormal, Beta, Gamma


prior_map = {
    Normal: lambda n, d: mc.Normal(n, mu=d.mu, sd=d.sd, shape=d.mu.shape),
    HalfNormal: lambda n, d: mc.HalfNormal(n, sd=d.sd, shape=d.sd.shape),
    Beta: lambda n, d: mc.Beta(n, alpha=d.alpha, beta=d.beta, shape=d.alpha.shape),
    Gamma: lambda n, d: mc.Gamma(n, alpha=d.alpha, beta=d.beta, shape=d.alpha.shape),
}


def construct_priors(nems_priors):
    '''
    Convert the NEMS priors to the format required by PyMC3.

    This conversion is pretty straightforward as the attribute names for NEMS
    priors are designed to map directly to the attributes on the corresponding
    PyMC3 prior class. The only reason I don't actually use PyMC3 priors (in
    lieu of our custom class) is to minimize dependencies on third-party
    libraries.

    If you don't want to do Bayes fitting or variational inference, then you
    don't need PyMC3.  However, priors are still extremely useful for non-Bayes
    fitters as they provide information about constraints.
    '''
    mc_priors = []
    for module_priors in nems_priors:
        module_mc_priors = {}
        for name, prior in module_priors.items():
            dtype = type(prior)
            converter = prior_map[dtype]
            module_mc_priors[name] = converter(name, prior)
        mc_priors.append(module_mc_priors)
    return mc_priors


def construct_bayes_model(nems_model, signals, prediction, observed,
                          batches=None):

    signals = signals.copy()
    nems_priors = nems_model.get_priors(signals)

    # Now, batch the signal if requested. The get_priors code typically doesn't
    # work with batched tensors.
    if batches is not None:
        for k, v in signals.items():
            signals[k] = mc.Minibatch(v, batch_size=batches)

    with mc.Model() as mc_model:
        mc_priors = construct_priors(nems_priors)
        tensors = nems_model.generate_tensor(signals, mc_priors)
        pred = tensors[prediction]
        obs = tensors[observed]
        likelihood = mc.Poisson('likelihood', mu=pred, observed=obs)
    return mc_model

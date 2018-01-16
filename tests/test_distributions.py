import pytest
import numpy as np
from scipy.stats import norm, beta
from nems.distributions.normal import Normal
from nems.distributions.half_normal import HalfNormal
import matplotlib.pyplot as plt


def test_distributions():
    
    # Two normal distributions
    d = Normal(mu=[-0.9, 0.2], sd=[1, 0.4])
    d2 = HalfNormal(sd=[1, 0.4])

    # Test plotting (TODO: how to make this a useful test?)
    # d.plot()
    # d2.plot()

    assert(100 == len(d.sample(100)))
    assert(100 == len(d2.sample(100)))


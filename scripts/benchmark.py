# A stripped down 'dummy fitter' that benchmarks

import os
import logging
import cProfile

from nems import initializers
from nems.fitters.api import dummy_fitter
from nems.analysis.api import fit_basic
from nems.recording import Recording

# -------------------------------------------------------------------------
# CONFIGURATION

logging.basicConfig(level=logging.WARN)

signals_dir = '../signals'
modelspecs_dir = '../modelspecs'

rec = Recording.load(os.path.join(signals_dir, 'gus027b13_p_PPS'))
est, val = rec.split_at_time(0.8)


@profile
def demo_benchmark(est):
    modelspec = initializers.from_keywords('wc40x1_fir10x1_dexp1')
    results = fit_basic(est, modelspec, fitter=dummy_fitter)

print('Benchmarking, please wait...')

for _ in range(3):
    demo_benchmark(est)

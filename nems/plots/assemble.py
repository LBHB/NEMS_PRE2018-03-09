import math
from functools import partial

import matplotlib.pyplot as plt


def freeze_defaults(plot_fns, recording, modelspec, evaluator):
    return [partial(pf, recording, modelspec, evaluator) for pf in plot_fns]

def simple_grid(partial_plots, nrows=1, figsize=(6,4)):
    nplots = len(partial_plots)

    fig = plt.figure(figsize=figsize)

    for i in range(nplots):
        col = math.ceil(i/nrows)
        plt.subplot(nrows, col, i)
        partial_plots[i]()

    return fig
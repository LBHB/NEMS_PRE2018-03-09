from functools import partial

import matplotlib.pyplot as plt


def freeze_defaults(plot_fns, recording, modelspec, evaluator):
    return [partial(pf, recording, modelspec, evaluator) for pf in plot_fns]

def simple_grid(partial_plots, nrows=1, ncols=1, figsize=(12,9)):
    nplots = len(partial_plots)
    fig = plt.figure(figsize=figsize)

    for i in range(nplots):
        plt.subplot(nrows, ncols, i+1)
        partial_plots[i]()

    return fig

def get_predictions(modelspecs, evaluator):
    """TODO. Given a list of modelspecs and an evaluator function,
    returns a list of predictions that can be plotted."""
    raise NotImplementedError


from functools import partial
import matplotlib.pyplot as plt
import nems.modelspec as ms

def freeze_defaults(plot_fns, recording, modelspec, evaluator):
    return [partial(pf, recording, modelspec, evaluator) for pf in plot_fns]

def simple_grid(partial_plots, nrows=1, ncols=1, figsize=(12,9)):
    nplots = len(partial_plots)
    fig = plt.figure(figsize=figsize)

    for i in range(nplots):
        plt.subplot(nrows, ncols, i+1)
        partial_plots[i]()

    return fig

def get_predictions(recording, modelspecs, evaluator):
    """TODO. Given a recording, a list of modelspecs, and an evaluator
    function, returns a list of prediction signals that can be plotted.
    """
    processed_recs = [ms.evaluate(recording, mspec) for mspec in modelspecs]
    predictions = [rec['pred'] for rec in processed_recs]
    return predictions

def get_modelspec_names(modelspecs):
    """Given a list of modelspecs, returns a list of descriptive names
    for identifying them in plots."""
    raise NotImplementedError
    metas = [m[0]['meta'] for m in modelspecs]
    names = [meta['name'] for meta in metas]
    return names

def quick_plot():
    raise NotImplementedError
    # TODO
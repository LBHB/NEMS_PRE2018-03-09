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

def get_predictions(recording, modelspecs, evaluator=ms.evaluate):
    '''
    Given a recording, a list of modelspecs, and optionally an evaluator function, 
    returns a list of prediction signals.
    '''
    recs = [evaluator(recording, mspec) for mspec in modelspecs]
    predictions = [rec['pred'] for rec in recs]
    return predictions

def get_modelspec_names(modelspecs):
    """Given a list of modelspecs, returns a list of descriptive names
    for identifying them in plots."""
    names = [ms.get_modelspec_name(m) for m in modelspecs]
    return names

def quick_plot():
    raise NotImplementedError
    # TODO

def plot_layout(plot_fn_struct):
    ''' 
    Accepts a list of lists of functions of 1 argument (ax). 
    Basically a fancy subplot that lets you lay out functions without
    worrying about details. See example below
    '''
    # Count how many plot functions we want
    nrows = len(plot_fn_struct)
    ncols = max([len(row) for row in plot_fn_struct])
    # Set up the subplots
    fig = plt.figure(figsize=(12,12))
    for r, row in enumerate(plot_fn_struct):
        for c, plotfn in enumerate(row):
            colspan = max(1, int(ncols / len(row)))
            ax = plt.subplot2grid((nrows, ncols), (r, c), colspan=colspan)  
            plotfn(ax=ax)
    return fig

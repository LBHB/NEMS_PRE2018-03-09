# -*- coding: utf-8 -*-
# wrapper code for fitting models

import os
import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import random
import numpy as np
import matplotlib.pyplot as plt
import nems
import nems.initializers
import nems.epoch as ep
import nems.priors
import nems.preprocessing as preproc
import nems.modelspec as ms
import nems.plots.api as nplt
import nems.metrics.api as nmetrics
import nems.analysis.api
import nems.utils
import nems.utilities.baphy as nb

from nems.recording import Recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize

def fit_model_baphy(cellid,batch,modelname,
                    autoPlot=True, saveInDB=False):

    """
    Fits a single NEMS model using data from baphy/celldb
    eg, 'ozgf100ch18_wc18x1_lvl1_fir15x1_dexp1_fit01'
    generates modelspec with 'wc18x1_lvl1_fir15x1_dexp1'

    """

    # parse modelname
    kws = modelname.split("_")
    loader = kws[0]
    modelspecname = "_".join(kws[1:-1])
    fitter = kws[-1]

    options = {}
    if loader == "ozgf100ch18":
        options["stimfmt"] = "ozgf"
        options["chancount"] = 18
        options["rasterfs"] = 100
    elif loader == "env100":
        options["stimfmt"] = "envelope"
        options["chancount"] = 0
        options["rasterfs"] = 100
    else:
        raise ValueError('unknown loader string')

    # set up data/output paths
    signals_dir = (
            "/auto/data/tmp/batch{0}_fs{1}_{2}{3}/{4}"
            .format(batch, options["rasterfs"], options["stimfmt"],
                    options["chancount"], cellid)
            )
    modelspecs_dir = '/auto/data/tmp/modelspecs'
    figures_dir = '/auto/data/tmp/figures'

    log.info('Loading data...')
    print(signals_dir)
    rec = Recording.load(signals_dir)

    # clone stim signal to create a placeholder for pred
    pred = rec['stim'].copy()
    pred.name = 'pred'
    rec.add_signal(pred)

    log.info('Withholding validation set data...')

    # Method #0: Try to guess which stimuli have the most reps, use those for val
    est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

    est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
    val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')

    log.info('Initializing modelspec(s)...')

    # Method #1: create from "shorthand" keyword string
    modelspec = nems.initializers.from_keywords(modelspecname)
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname}
    if not 'meta' in modelspec[0].keys():
        modelspec[0]['meta'] = meta
    else:
        modelspec[0]['meta'].update(meta)
    # changed to above so that meta doesn't get removed if it already exists
    # Can remove the 4 lines below if this doesn't cause any issues
    #     -jacob 2-25-18
    #modelspec[0]['meta'] = {}
    #modelspec[0]['meta']['batch'] = batch
    #modelspec[0]['meta']['cellid'] = cellid
    #modelspec[0]['meta']['modelname'] = modelname

    log.info('Fitting modelspec(s)...')

    # Option 1: Use gradient descent on whole data set(Fast)
    if fitter == "fit01":
        # prefit strf
        log.info("Prefitting STRF without other modules...")
        modelspec = nems.initializers.prefit_to_target(
                est, modelspec, nems.analysis.api.fit_basic, 'fir_filter',
                fitter=scipy_minimize,
                fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}}
                )
        log.info("Performing full fit...")
        modelspecs = nems.analysis.api.fit_basic(est, modelspec,
                                                 fitter=scipy_minimize)
    else:
        raise ValueError('unknown fitter string')

    log.info('Saving modelspec(s) ...')
    ms.save_modelspecs(modelspecs_dir, modelspecs)

    log.info('Generating summary statistics...')
    # TODO

    new_rec = [ms.evaluate(val, m) for m in modelspecs]
    cc = [nmetrics.corrcoef(p, 'pred', 'resp') for p in new_rec]
    print(cc)

    #log.info("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],

    if autoPlot:
        # GENERATE PLOTS
        log.info('Generating summary plot...')

        # Generate a summary plot
        fig = nplt.plot_summary(val, modelspecs)
        figpath = nplt.save_figure(fig, modelspecs=modelspecs,
                                   save_dir=figures_dir)
        # Pause before quitting
        plt.show()

    # save in database
    if saveInDB:
        #filename = nems.utilities.io.get_file_name(cellid, batch, modelname)
        pass

    return(modelspecs)


def examine_recording(rec, epoch_regex='TRIAL', occurrence=0):
    # plot example spectrogram and psth from one trial
    # todo: regex matching (currently just does exatract string matching)
    # interactive?
    #
    stim = rec['stim']
    resp = rec['resp']

    plt.figure()
    ax = plt.subplot(2,1,1)
    if stim.nchans>2:
        nplt.spectrogram_from_epoch(stim, epoch_regex, ax=ax, occurrence=occurrence)
    else:
        nplt.timeseries_from_epoch([stim], epoch_regex, ax=ax, occurrence=occurrence)
    plt.title("{0} # {1}".format(epoch_regex,occurrence))
    ax = plt.subplot(2,1,2)
    nplt.timeseries_from_epoch([resp], epoch_regex, ax=ax, occurrence=occurrence)

    plt.tight_layout()



cellid = 'chn020f-b1'
batch = 271
modelname = "ozgf100ch18_wc18x1_lvl1_fir15x1_dexp1_fit01"

#cellid='btn144a-c1'
#batch=259
#modelname="env100_fir15x2_dexp1_fit01"


modelspecs = fit_model_baphy(cellid,batch,modelname)


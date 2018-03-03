# -*- coding: utf-8 -*-
# wrapper code for fitting models

import os

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
import nems.metrics.api
import nems.analysis.api
import nems.utils
import nems.utilities.baphy as nb
import nems.db as nd

from nems.recording import Recording
from nems.fitters.api import dummy_fitter, coordinate_descent, scipy_minimize

import logging
log = logging.getLogger(__name__)
#logging.basicConfig(level=logging.INFO)

def run_loader_baphy(cellid,batch,loader):
    options = {}
    if loader == "ozgf100ch18":
        options["stimfmt"] = "ozgf"
        options["chancount"] = 18
        options["rasterfs"] = 100
        options["average_stim"]=True
        options["state_vars"]=False
    elif loader == "ozgf100ch18pup":
        options["stimfmt"] = "ozgf"
        options["chancount"] = 18
        options["rasterfs"] = 100
        options["average_stim"]=False
        options["state_vars"]=['pupil']
    elif loader == "env100":
        options["stimfmt"] = "envelope"
        options["chancount"] = 0
        options["rasterfs"] = 100
        options["average_stim"]=True
        options["state_vars"]=False
    else:
        raise ValueError('unknown loader string')
        
    url="http://potoroo:3003/baphy/{0}/{1}?fs={2}&stimfmt={3}&chancount={4}".format(
            batch, cellid, options["rasterfs"], options["stimfmt"],
            options["chancount"])
    # set up data/output paths
    signals_dir = (
            "/auto/data/tmp/batch{0}_fs{1}_{2}{3}/{4}"
            .format(batch, options["rasterfs"], options["stimfmt"],
                    options["chancount"], cellid)
            )

    log.info('Loading {0} format for {1}/{2}...'.format(loader,cellid,batch))
    rec = Recording.load(signals_dir)

    # before this can be replaced with 
    """
    URL = "http://potoroo:3004/baphy/271/bbl086b-11-1?rasterfs=200"
    rec = Recording.load_url(URL)
    """
    
    if options["state_vars"]:
        rec=preproc.make_state_signal(rec, state_signals=options["state_vars"])

    # Method #0: Try to guess which stimuli have the most reps, use those for val
    if options["average_stim"]:
        log.info('Withholding validation set data...')
        est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')
        
        est = preproc.average_away_epoch_occurrences(est, epoch_regex='^STIM_')
        val = preproc.average_away_epoch_occurrences(val, epoch_regex='^STIM_')
    else:
        #preserve single trial data, no val set... for nested CV
        est=rec
        val=rec.copy()
        
    # TO DO : Leave open option to have different est val splits (or none)

    return est,val


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
    
    est,val = run_loader_baphy(cellid,batch,loader)
    
    modelspecs_dir = '/auto/data/tmp/modelspecs/{0}/{1}'.format(batch,cellid)
    figures_dir = modelspecs_dir
    
    log.info('Initializing modelspec(s) for cell/batch {0}/{1}...'.format(cellid,batch))

    # Method #1: create from "shorthand" keyword string
    modelspec = nems.initializers.from_keywords(modelspecname)
    if 'CODEHASH' in os.environ.keys():
        codehash=os.environ['CODEHASH']
    else:
        codehash=""
    meta = {'batch': batch, 'cellid': cellid, 'modelname': modelname,
            'loader': loader, 'fitter': fitter, 'modelspecname': modelspecname,
            'username': 'svd', 'labgroup': 'lbhb', 'public': 1, 
            'codehash': codehash}
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
                est, modelspec, nems.analysis.api.fit_basic, 'levelshift',
                fitter=scipy_minimize,
                fit_kwargs={'options': {'ftol': 1e-4, 'maxiter': 500}}
                )
        log.info("Performing full fit...")
        modelspecs = nems.analysis.api.fit_basic(est, modelspec,
                                                 fitter=scipy_minimize)
    else:
        raise ValueError('unknown fitter string')

    log.info('Generating summary statistics...')
    # TODO

    new_rec = [ms.evaluate(val, m) for m in modelspecs]
    r_test = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_rec]
    new_rec = [ms.evaluate(est, m) for m in modelspecs]
    r_fit = [nems.metrics.api.corrcoef(p, 'pred', 'resp') for p in new_rec]
    modelspecs[0][0]['meta']['r_fit']=np.mean(r_fit)
    modelspecs[0][0]['meta']['r_test']=np.mean(r_test)
    
    log.info("r_fit={0} r_test={1}".format(modelspecs[0][0]['meta']['r_fit'],
          modelspecs[0][0]['meta']['r_test']))
    print("r_fit={0} r_test={1}".format(modelspecs[0][0]['meta']['r_fit'],
          modelspecs[0][0]['meta']['r_test']))
    
    savepath=ms.save_modelspecs(modelspecs_dir, modelspecs)
    log.info('Saved modelspec(s) to {0} ...'.format(savepath))
    
    if autoPlot:
        # GENERATE PLOTS
        log.info('Generating summary plot...')

        # Generate a summary plot
        fig = nplt.plot_summary(val, modelspecs)
        figpath = nplt.save_figure(fig, modelspecs=modelspecs,
                                   save_dir=figures_dir)
        # Pause before quitting
        plt.show()
        
        modelspecs[0][0]['meta']['figurefile']=figpath
        
    # save in database
    if saveInDB:
        
        # TODO : db results
        
        logging.info('Saved results to {0}'.format(savepath))
        modelspecs[0][0]['meta']['modelpath']=savepath

        nd.update_results_table(modelspecs[0])
        return savepath
    else:
        
        return modelspecs


def load_model_baphy(filepath,loadrec=True):

    logging.info('Loading modelspecs...')

    modelspec=ms.load_modelspec(filepath)

    cellid=modelspec[0]['meta']['cellid']
    batch=modelspec[0]['meta']['batch']
    loader=modelspec[0]['meta']['loader']

    if loadrec:
        est,val = run_loader_baphy(cellid,batch,loader)
        
        return modelspec,est,val
    else:
        return modelspec
    
    
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


def fit_batch(batch, modelname="ozgf100ch18_wc18x1_lvl1_fir15x1_dexp1_fit01"):
    plt.close('all')
    cell_data=nd.get_batch_cells(batch=batch)
    cellids=list(cell_data['cellid'].unique())
    
    for cellid in cellids:
        fit_model_baphy(cellid,batch,modelname, autoPlot=True, saveInDB=True)
        
    
def quick_inspect(cellid="chn020f-b1", batch=271, 
               modelname="ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"):
    d=nd.get_results_file(batch,[modelname],[cellid])
    savepath=d['modelpath'][0]
    modelspec,est,val=load_model_baphy(savepath)
    fig = nplt.plot_summary(val, [modelspec])
    
    return modelspec,est,val

"""
# SPN example
cellid='btn144a-c1'
batch=259
modelname="env100_fir15x2_dexp1_fit01"

# A1 NAT example
cellid = 'TAR010c-18-1'
batch=271
modelname = "ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"

# A1 NAT + pupil example
cellid = 'TAR010c-18-1'
batch=289
modelname = "ozgf100ch18_wcg18x2_fir15x2_lvl1_dexp1_fit01"

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)

# IC NAT example
cellid = "bbl031f-a1"
batch=291
modelname = "ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01"

savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
modelspec,est,val=load_model_baphy(savepath)

"""

#plt.close('all')

#cellid = "bbl034e-a1"
#batch=291
#modelname = "ozgf100ch18_dlog_wc18x1_fir15x1_lvl1_dexp1_fit01"

# this does now work:
#savepath = fit_model_baphy(cellid = cellid, batch=batch, modelname = modelname, autoPlot=True, saveInDB=True)
#modelspec,est,val=load_model_baphy(savepath)

#modelspec,est,val=quick_inspect("bbl036e-a2",291,"ozgf100ch18_wc18x1_fir15x1_lvl1_dexp1_fit01")

# this works the first time you run
#savepath = fit_model_baphy(cellid= 'chn020f-b1',batch=batch,modelname=modelname, autoPlot=True, saveInDB=True)

# what I'd like to be able to run:
#fit_batch(batch,modelname)

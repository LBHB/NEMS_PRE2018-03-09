"""View functions for "Fit Single Now" and "Enqueue Models" buttons.

These functions communicate with modelfit.py and are called by flask 
when the browser navigates to their app.route URL signatures. 
fit_single_model_view calls modelfit.fit_single_model
for the cell, batch and model selection passed via AJAX.
enqueue_models_view calls enqueue_models for the list of
cell selections, batch selection, and list of model selections
passed via AJAX.
Both functions return json-serialized summary information for the user
to indicate the success/failure and results of the model fit/queue.

See Also:
---------
. : modelfit.py

"""

from itertools import product

from flask import render_template, jsonify, request
import pandas as pd
import matplotlib.pyplot as plt, mpld3

from nems_analysis import app, NarfResults, Session
#from model_functions.modelfit import fit_single_model, enqueue_models
import lib.nems_utils as nu
import lib.nems_modules as nm
import lib.nems_keywords as nk
import lib.nems_fitters as nf
import lib.nems_main as nems
from model_functions.fit_single_utils import fetch_meta_data


@app.route('/fit_single_model')
def fit_single_model_view():
    """Call lib.nems_main.fit_single_model with user selections as args."""
    
    session = Session()
    
    cSelected = request.args.getlist('cSelected[]')
    bSelected = request.args.get('bSelected')[:3]
    mSelected = request.args.getlist('mSelected[]')

    # Disallow multiple cell/model selections for a single fit.
    if (len(cSelected) > 1) or (len(mSelected) > 1):
        return jsonify(r_est='error',r_val='more than 1 cell and/or model')
        
    print(
            "Beginning model fit -- this may take several minutes.",
            "Please wait for a success/failure response.",
            )
    try:
        stack = nems.fit_single_model(
                cellid=cSelected[0], batch=bSelected, modelname=mSelected[0],
                autoplot=False,
                )
    except Exception as e:
        print("Error when calling nems_main.fit_single_model")
        print(e)
        print("Fit failed.")
        raise e
        
    plotfile = stack.quick_plot_save(mode="png")

    r = (
            session.query(NarfResults)
            .filter(NarfResults.cellid == cSelected[0])
            .filter(NarfResults.batch == bSelected)
            .filter(NarfResults.modelname == mSelected[0])
            .all()
            )
    collist = ['%s'%(s) for s in NarfResults.__table__.columns]
    attrs = [s.replace('NarfResults.', '') for s in collist]
    attrs.remove('id')
    attrs.remove('figurefile')
    attrs.remove('lastmod')
    if not r:
        r = NarfResults()
        r.figurefile = plotfile
        fetch_meta_data(stack, r, attrs)
        # TODO: assign performance variables from stack.meta
        session.add(r)
    else:
        # TODO: assign performance variables from stack.meta
        r[0].figurefile = plotfile
        fetch_meta_data(stack, r[0], attrs)
    
    session.commit()
    session.close()
    
    return jsonify(r_est=stack.meta['r_est'][0], r_val=stack.meta['r_val'][0])

@app.route('/enqueue_models')
def enqueue_models_view():
    """Call modelfit.enqueue_models with user selections as args."""
    
    session = Session()
    #max number of models to run?
    queuelimit = request.args.get('queuelimit')
    if queuelimit:
        queuelimit = int(queuelimit)
    
    # Only pull the numerals from the batch string, leave off the description.
    bSelected = request.args.get('bSelected')[:3]
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    
    # TODO: What should this return? What does the user need to see?
    # data = enqueue_models(celllist=cSelected,batch=bSelected,
    #                      modellist=mSelected)
    
    
    combos = list(product(cSelected, mSelected))
    failures = []
    for combo in combos[:queuelimit]:
        cell = combo[0]
        model = combo[1]
        try:
            stack = nems.fit_single_model(
                    cellid=cell, batch=bSelected,
                    modelname=model, autoplot=False,
                    )
        except Exception as e:
            print("Error when calling nems.fit_single_model for " + mSelected)
            print(e)
            failures += combo
            continue
        plotfile = stack.quick_plot_save()
        r = (
                session.query(NarfResults)
                .filter(NarfResults.cellid == cell)
                .filter(NarfResults.batch == bSelected)
                .filter(NarfResults.modelname == model)
                .all()
                )
        collist = ['%s'%(s) for s in NarfResults.__table__.columns]
        attrs = [s.replace('NarfResults.', '') for s in collist]
        attrs.remove('id')
        attrs.remove('figurefile')
        attrs.remove('lastmod')
        if not r:
            r = NarfResults()
            r.figurefile = plotfile
            fetch_meta_data(stack, r, attrs)
            # TODO: assign performance variables from stack.meta
            session.add(r)
        else:
            # TODO: assign performance variables from stack.meta
            r[0].figurefile = plotfile
            fetch_meta_data(stack, r[0], attrs)

        session.commit()

    session.close()
    
    if queuelimit and (queuelimit >= len(combos)):
        data = (
                "Queue limit present. The first %d "
                "cell/model combinations have been fitted (all)."%queuelimit
                )
    elif queuelimit and (queuelimit < len(combos)):
        data = (
                "Queue limit exceeded. Some cell/model combinations were "
                "not fit (%d out of %d fit)."%(queuelimit, len(combos))
                )
    else:
        data = "All cell/model combinations have been fit (no limit)."
        
    if failures:
        failures = ["%s, %s\n"%(c[0],c[1]) for c in failures]
        failures = " ".join(failures)
        data += "\n Failed combinations: %s"%failures
    
    return jsonify(data=data)
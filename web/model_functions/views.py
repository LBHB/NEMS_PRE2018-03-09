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

from flask import render_template, jsonify, request
import pandas as pd

from nems_analysis import app, NarfResults, Session
#from model_functions.modelfit import fit_single_model, enqueue_models
import lib.nems_utils as nu
import lib.nems_modules as nm
import lib.nems_keywords as nk
import lib.nems_fitters as nf
import lib.nems_main as nems
import model_functions.fit_single_utils as fsu


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
    
    stack = nems.fit_single_model(
            cellid=cSelected[0], batch=bSelected, modelname=mSelected[0],
                            )
    
    plotfile = nu.quick_plot_save(stack, mode="json")
    
    r = (
            session.query(NarfResults)
            .filter(NarfResults.cellid == cSelected[0])
            .filter(NarfResults.batch == bSelected)
            .filter(NarfResults.modelname == mSelected[0])
            .all()
            )
    
    if not r:
        r = NarfResults()
        r.cellid = cSelected[0]
        r.batch = bSelected
        r.modelname = mSelected[0]
        r.figurefile = plotfile
        # TODO: assign performance variables from stack.meta
        session.add(r)
    else:
        # TODO: assign performance variables from stack.meta
        r.figurefile = plotfile
        pass
    
    session.commit()
    session.close()
    
    return jsonify(r_est=stack.meta['r_est'][0], r_val=stack.meta['r_val'][0])

#@app.route('/enqueue_models')
def enqueue_models_view():
    """Call modelfit.enqueue_models with user selections as args."""
    
    # Only pull the numerals from the batch string, leave off the description.
    bSelected = request.args.get('bSelected')[:3]
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    
    # TODO: What should this return? What does the user need to see?
    data = enqueue_models(celllist=cSelected,batch=bSelected,
                          modellist=mSelected)
    
    return jsonify(data=data,testb=bSelected,testc=cSelected,testm=mSelected)
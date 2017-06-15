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

from nems_analysis import app
from model_functions.modelfit import fit_single_model, enqueue_models


@app.route('/fit_single_model')
def fit_single_model_view():
    """Call modelfit.fit_single_model with user selections as args."""
    
    cSelected = request.args.getlist('cSelected[]')
    # Only pull the numerals from the batch string, leave off the description
    bSelected = request.args.get('bSelected')[:3]
    mSelected = request.args.getlist('mSelected[]')
    
    # Disallow multiple cell/model selections for a single fit.
    if (len(cSelected) > 1) or (len(mSelected) > 1):
        return jsonify(data='error',preview='more than 1 cell and/or model')
    
    # TODO: What should this return?
    #       Only need to return data that will be displayed to user,
    #       not the full data array from the model fitter.
    #       --figurefile/preview image and what else?
    data = fit_single_model(cellid=cSelected[0],batch=bSelected,
                            modelname=mSelected[0])
    
    # TODO: how will data be formatted?
    #figure_file = data['figure_file']
    figure_file = 'preview for %s, %s, %s'\
                  %(cSelected[0],bSelected,mSelected[0])
    
    return jsonify(data = data.data,preview = figure_file)

@app.route('/enqueue_models')
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
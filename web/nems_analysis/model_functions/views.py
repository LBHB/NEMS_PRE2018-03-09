from flask import render_template, jsonify, request
from nems_analysis import app
import pandas as pd
from model_functions.modelfit import fit_single_model, enqueue_models

@app.route('/modelpane')
def modelpane_view():
    return render_template('/modelpane/modelpane.html')

@app.route('/fit_single_model')
def fit_single_model_view():
    #TODO: link up to model package
    cSelected = request.args.getlist('cSelected[]')
    bSelected = request.args.get('bSelected')
    mSelected = request.args.getlist('mSelected[]')
    
    if (len(cSelected) > 1) or (len(mSelected) > 1):
        return jsonify(data ='error',preview='more than 1 cell and/or model')
    
    # get cellid and modelname from results table selection,
    # batch from batch selector
    
    # TODO: what do I want to have returned from this?
    #       model queue should only have to run this one line, so only need to 
    #       return things that will be displayed to user on single fit
    #       --figurefile/preview image and what else?
    #       --nothing displayed for enqueue since they don't run right away?
    data = fit_single_model(cellid=cSelected[0],batch=bSelected,modelname=mSelected[0])

    #figure_file = data['figure_file']
    figure_file = 'preview for %s, %s, %s'%(cSelected[0],bSelected,mSelected[0])
    
    #use data after ajax call to display some type of results summary or success message?
    return jsonify(data = data,preview = figure_file)

@app.route('/enqueue_models')
def enqueue_models_view():
    
    #TODO: is batch necessary? cell list has already been retrieved, but
    #       might still need to record batch # in NarfResults at the  end.
    bSelected = request.args.get('bSelected')[:3]
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    
    # get batch, cell and model selections from their respective selectors.
    # only queue models for selections, not entire analysis
    data = enqueue_models(celllist=cSelected,batch=bSelected,modellist=mSelected)
    
    # only need to return some kind of success/failure message or summary
    # fitting doesn't happen right away so no image or statistics to return
    return jsonify(data = data, testb=bSelected,testc=cSelected,testm=mSelected)
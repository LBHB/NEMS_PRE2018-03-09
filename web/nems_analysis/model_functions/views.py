from flask import render_template, jsonify, request, Response
from nems_analysis import app, Session, NarfAnalysis, NarfBatches, NarfResults
import pandas.io.sql as psql
import pandas as pd

@app.route('/modelpane')
def modelpane_view():
    return render_template('/modelpane/modelpane.html')

@app.route('/single_fit')
def single_fit():
    #TODO: link up to model package
    
    session = Session()
    
    cSelected = request.args.get('cSelected')
    bSelected = request.args.get('bSelected')
    mSelected = request.args.get('mSelected')
    
    #import model fitter object from model package at top
    #ModelObject = ModelFitterObjectName\
                    #(cellid=cSelected,batch=bSelected,model=mSelected)
    
    #data_from_modules = ModelObject.methodForRetrievingData()
    
    #will be 3-d arrays? may need to convert objects in frame to series, or use panel
    
    #dataframe = pd.Dataframe(data_from_modules,index=someIndex)
    #or put each array into separate dataframe, whatever works better
    
    """
    session.add_all([
            Table(column1=data_from_modules['column1'], column2=data_from_modules['column2']),
            --repeat for each entry
            //construct list outside of this first for multiple entries
            ])
    
    session.commit()
    """
    
    # copy code from nems_analysis/views for @get_preview
    # or call function from ajax function after retrieving data
    figure_file = 'summary image .png'
    
    session.close()
    return jsonify(data = data_from_modules)

@app.route('/enqueue_models')
def enqueue_models():
    
    return Response('Placeholder for model queueing function')
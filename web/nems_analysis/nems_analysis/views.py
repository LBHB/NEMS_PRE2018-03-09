from flask import render_template, jsonify, request
import nems_analysis
from nems_analysis import app
from nems_analysis.ModelFinder import ModelFinder
import pandas.io.sql as psql

@app.route('/')
def main_view():
    # hard coded for now, add more options later
    plottypelist = 'Scatter'

    analysislist = nems_analysis.analyses['name'].tolist()
    batchlist = nems_analysis.batches.iloc[:,0].tolist()
    
    #testing dataframes
    
    return render_template('main.html',analysislist = analysislist,\
                           batchlist = batchlist,\
                           plottypelist = plottypelist,\
                           )
    
@app.route('/update_batch')
def update_batch():
    aSelected = request.args.get('aSelected',type=str)
    a = nems_analysis.analyses
    batch = a.loc[a['name'] == aSelected, 'batch'].iloc[0]
    
    return jsonify(batch = batch)
    

@app.route('/update_models')
def update_models():
    aSelected = request.args.get('aSelected',type=str)
    a = nems_analysis.analyses
    modeltree = a.loc[a['name'] == aSelected, 'modeltree'].iloc[0]
                
    mf = ModelFinder(modeltree)
    modellist = mf.modellist
    
    return jsonify(modellist=modellist)

@app.route('/update_cells')
def update_cells():
    #just use first 3 indices of str to get the integer-only representation
    bSelected = request.args.get('bSelected',type=str)[:3]
    c = nems_analysis.cells
    celllist = c.loc[c['batch'] == bSelected, 'cellid'].tolist()
               
    return jsonify(celllist=celllist)

@app.route('/update_results')
def update_results():
    # TODO: Change Jquery for this function to update on button click
    # instead of on selection change.
    
    bSelected = request.args.get('bSelected')
    if len(bSelected) == 0:
        return jsonify(resultstable='MUST SELECT A BATCH')
    bSelected = int(bSelected[:3])
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')
    
    #TODO: figure out a better way to do this. may have to go back to 
    #generating query with for loops based on case.
    #or try with sql alchemy expressions
    r = psql.read_sql('SELECT * FROM NarfResults WHERE batch=%d LIMIT 10000'\
                        %bSelected, nems_analysis.engine)
    
    # no cells or models selected
    if (len(cSelected) == 0) and (len(mSelected) == 0):
        # TODO: only shows first 500 results -- code this in as a user option
        #       that can be adjusted near the table display
        results = r[r.batch == bSelected].head(500)
    # no cells selected, but model(s) selected
    elif (len(cSelected) == 0) and (len(mSelected) != 0):
        results = r[(r.batch == bSelected) & (r.modelname.isin(mSelected))]
    # cell(s) selected, but no model(s) selected
    elif (len(cSelected) != 0) and (len(mSelected) == 0):
        results = r[(r.batch == bSelected) & (r.cellid.isin(cSelected))]
    # both cell(s) and model(s) selected
    else:
        results = r[(r.batch == bSelected) & (r.modelname.isin(mSelected))\
                    & (r.cellid.isin(cSelected))]
        
    return jsonify(resultstable=results.to_html(classes='table-hover\
                                                table-condensed'))

@app.route('/make_plot', methods=['POST'])
def make_plot():
    pass

@app.route('/scatter_plot', methods=['GET','POST'])
def scatter_plot():
    pass

@app.route('/empty')
def empty_plot():
    pass

@app.route('/error_log')
def error_log():
    pass

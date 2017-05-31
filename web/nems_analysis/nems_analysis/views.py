from flask import render_template, jsonify, request
import nems_analysis
from nems_analysis import app, session, NarfAnalysis, NarfBatches, NarfResults
from nems_analysis.ModelFinder import ModelFinder
import pandas.io.sql as psql



##################################################################
####################   UI UPDATE FUNCTIONS  ######################
##################################################################


# landing page
@app.route('/')
def main_view():
    # hard coded for now, add more options later
    plottypelist = 'Scatter'

    # .all() returns a list of tuples, list comprehension to pull tuple
    # elements out into list
    analysislist = [i[0] for i in session.query(NarfAnalysis.name).all()]
    batchlist = [i[0] for i in session.query(NarfAnalysis.batch).distinct().all()]
    
    #testing dataframes
    
    return render_template('main.html',analysislist = analysislist,\
                           batchlist = batchlist,\
                           plottypelist = plottypelist,\
                           )
    
@app.route('/update_batch')
def update_batch():
    aSelected = request.args.get('aSelected',type=str)
    
    batch = session.query(NarfAnalysis.batch).filter\
            (NarfAnalysis.name == aSelected).first()[0]
    
    return jsonify(batch = batch)
    

@app.route('/update_models')
def update_models():
    aSelected = request.args.get('aSelected',type=str)
    
    modeltree = session.query(NarfAnalysis.modeltree).filter\
                (NarfAnalysis.name == aSelected).first()[0]
                
    mf = ModelFinder(modeltree)
    modellist = mf.modellist
    
    return jsonify(modellist=modellist)

@app.route('/update_cells')
def update_cells():
    #just use first 3 indices of str to get the integer-only representation
    bSelected = request.args.get('bSelected',type=str)[:3]

    celllist = [i[0] for i in session.query(NarfBatches.cellid).filter\
               (NarfBatches.batch == bSelected).all()]
               
    return jsonify(celllist=celllist)

@app.route('/update_results')
def update_results():
    # TODO: Change Jquery for this function to update on button click
    # instead of on selection change.
    nullselection = 'MUST SELECT A BATCH AND ONE OR MORE CELLS AND ONE OR MORE\
                    MODELS BEFORE RESULTS WILL UPDATE'
    
    bSelected = request.args.get('bSelected')
    cSelected = request.args.getlist('cSelected[]')
    mSelected = request.args.getlist('mSelected[]')  
    if (len(bSelected) == 0) or (len(cSelected) == 0) or (len(mSelected) == 0):
        return jsonify(resultstable=nullselection)
    
    # TODO: only shows first 500 results -- code this in as a user option
    #       that can be adjusted near the table display for larger selections
    

    results = psql.read_sql_query(session.query(NarfResults).filter\
              (NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).limit(500).statement,\
              session.bind)
        
    return jsonify(resultstable=results.to_html(classes='table-hover\
                                                table-condensed'))


####################################################################
####################    PLOT FUNCTIONS    ##########################
####################################################################


# TODO: May want to split these up into a separate 'plot' package with
#       its own folder and views file as options grow

@app.route('/scatter_plot', methods=['GET','POST'])
def scatter_plot():
    pass

@app.route('/empty')
def empty_plot():
    pass



####################################################################
###################     MISCELLANEOUS  #############################
####################################################################


# clicking error log link will open text file with notes
# TODO: add interface to edit text from site, or submit notes some other way,
# so that users can report bugs/undesired behavior
@app.route('/error_log')
def error_log():
    return app.send_static_file('error_log.txt')

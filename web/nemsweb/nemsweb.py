"""
- also acting as __init__.py since couldn't get package setup to work
- may need to revisit re-organizing as package in the future, but continuing
- as app for now

- views functions - controls flow of flask app based on URL
- whenever user navigates to new URL generated by pain.py,
- views looks for the corresponding @app.route to decide which
- function to run, and parses any variables within the URL
- to be passed to the proper helper object (i.e. query or plot generator)
"""

from flask import Flask,render_template, redirect, request, jsonify, url_for
import pandas as pd
import QueryGenerator as qg
import PlotGenerator as pg
import DB_Connection as dbcon
import ModelFinder as mf

app = Flask(__name__)
app.config.from_object('config')
# TODO: re-write app.routes using new object structure

# create a database connection, then assign it to
# dbc to be passed to other objects as needed
db = dbcon.DB_Connection()
db.connect_lab()
dbc = db.connection
# use connect_lab() at lab, connect_remote() for amazon server


@app.route("/")
def main_view():
    # hard coded as only option for now - not really needed for plots,
    # only needed if want to look at tables in adatabase type fashion
    tablelist = 'NarfResults'
    plottypelist = 'Scatter'
    measurelist = 'r_test'
    # TODO: currently queries entire table, then picks out batch ids and model names
    # should figure out a better way to do this
    # i.e. don't query anything until user makes a selection
    # read entire celldb one time to create text file lists?
    # then only have to update when new data added, otherwise
    # can pull lists of batch nums, model names etc from text file instead.
    analyses = qg.QueryGenerator(dbc,column='name',tablename='NarfAnalysis').send_query()
    analyses = analyses.iloc[:,0] #convert from 1-dim df to series
    analysislist = analyses.tolist()

    batches = qg.QueryGenerator(dbc,column='batch',tablename='NarfBatches').send_query()
    batches = batches.iloc[:,0]
    # get unique list since there are duplicates
    batchlist = list(set(batches.tolist()))
    
    ## Maybe don't need this one with new setup? Cells can just populate after
    ## analysis is selected. Unless want to be able to sort by a specific cell first.
    cells = qg.QueryGenerator(dbc,column='cellid',tablename='NarfBatches').send_query()
    cells = cells.iloc[:,0]
    # get unique list since there are duplicates
    celllist = list(set(cells.tolist()))
    
    models = qg.QueryGenerator(dbc,column='modelname',tablename='NarfResults').send_query()
    models = models.iloc[:,0]
    modellist = list(set(models.tolist()))
    
    return render_template('main.html', analysislist = analysislist,\
                           tablelist = tablelist,\
                           batchlist = batchlist,\
                           celllist = celllist,\
                           modellist = modellist,\
                           plottypelist = plottypelist,
                           measurelist = measurelist,
                           )


########     UI update handlers for main page     #########
###########################################################


# TODO: implement this functionality -- should update batch, model etc based
#       on selected analysis

# update batch option based on analysis selection
@app.route("/update_batch")
def update_batch():
    aSelected = request.args.get('aSelected')
    analysis = qg.QueryGenerator(dbc,tablename='NarfAnalysis',\
                                 analysis=aSelected).send_query()
    batchnum = analysis.get_value(analysis.index.values[0],'batch')
    
    #return batchnum for selected analysis in jQuery-friendly format
    return jsonify(batchnum)
    
# could probably put both of these together to reduce number of queries, 
# but leaving separate for now to make jquery code clearer

# update model list based on analysis selection
@app.route("/update_models")
def update_models():
    aSelected = request.args.get('aSelected')
    analysis = qg.QueryGenerator(dbc,tablename='NarfAnalysis',\
                                 analysis=aSelected).send_query()
    
    # pull modeltree text from NarfAnalysis
    # and convert to string rep
    modeltree = analysis['modeltree'][0]
    modelFinder = mf.ModelFinder(modeltree)
    modellist = modelFinder.modellist
    
    # TODO: figure out how to turn analysis modelstring into new model list
    # re-code keyword_combos from narf in python?
    return jsonify(modellist)
    
    
# update cell list based on batch selection
@app.route("/update_cells")
def update_cells():
    bSelected = request.args.get('bSelected')
    celllist = qg.QueryGenerator(dbc,tablename='NarfBatches', batch=bSelected)
    
    return jsonify(celllist)
    
    
    
##########       TABLES        ############
###########################################

# takes form data submitted via select objects at '/'
# and uses it to form url for req_query
@app.route("/view_database")
def view_database():
    tablelist = 'NarfResults'
    
    batches = qg.QueryGenerator(dbc,column='batch',tablename='NarfBatches').send_query()
    batches = batches.iloc[:,0]
    # get unique list since there are duplicates
    batchlist = list(set(batches.tolist()))
    
    models = qg.QueryGenerator(dbc,column='modelname',tablename='NarfResults').send_query()
    models = models.iloc[:,0]
    modellist = list(set(models.tolist()))
    
    return render_template('database.html', tablelist=tablelist, batchlist=batchlist,\
                           modellist=modellist)

@app.route("/req_query", methods = ['POST'])
def req_query():
    # add if statements to check if request data exists before pulling
    tablename = request.form['tablename']
    batchnum = request.form['batchnum']
    modelname = request.form['modelname']
    
    query = qg.QueryGenerator(dbc,tablename=tablename,batchnum=batchnum,\
                              modelname=modelname)
    # populate dataframe by calling send_query() on qg object
    data = query.send_query()

    tabletitle = ("%s, filter by: batch=%s, model=%s" %(tablename, batchnum, modelname))
    
    # generage html page via table.html template, pass in html export of dataframe
    # and table title as variables
    return render_template('table.html', table=data.to_html(classes='Table'),\
                           title=tabletitle)



###########       PLOTS       #############
###########################################

# take form data submitted at '/'
# and pass to generate_plot function
# used for generic plot generator - may change to individual routes?
@app.route("/make_plot", methods=['POST'])
def make_plot():
    plottype = request.form['plottype']
    tablename = request.form['tablename']
    batchnum = request.form['batchnum']
    # TODO: using these names for now since scatter plot only option,
    # but will want new naming scheme when more options added
    modelnameX = request.form['modelnameX']
    modelnameY = request.form['modelnameY']
    measure = request.form['measure']
    
    plot = pg.PlotGenerator(dbc, plottype, tablename, batchnum,\
                            modelnameX, modelnameY, measure)
    
    return render_template("plot.html", script=plot.plot[0], div=plot.plot[1])
    

# use form data to make scatter plot
# use one of these for each plot type instead of generic above?
@app.route("/scatter_plot", methods=['GET','POST'])
def scatter_plot():
    batchnum = request.form['batchnum']
    modelnames = request.form.getlist('modelnames')
    measure = request.form['measure']
    
    plot = pg.PlotGenerator(dbc,plottype='Scatter',batchnum=batchnum,\
                            modelnames=modelnames,measure=measure)
    plot.scatter_plot()
    
    return render_template("plot.html", script=plot.plot[0], div=plot.plot[1])

    
# check for empty plot request
# uncomment data.size==0 check in PlotGenerator.py to enable
@app.route("/empty")
def empty_plot():
    return "Empty plot, sad face, try again! If you're seeing this, the \
            cell list query returned no results"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug='True')

#disconnect from database when app shuts down
@app.teardown_appcontext
def disconnect_from_db():
    db.close_connection()

 
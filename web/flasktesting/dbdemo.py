#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:36:59 2017

@author: jacob
"""
import numpy as np
import pandas.io.sql as psql
import pymysql as pysql
import scipy as scpy
from flask import *
import pandas as pd
from connectfunc import *

app = Flask(__name__)

#currently only have one user input partially working (dropbox on /index page)
#all other funcitonality accessed directly in address bar based on @approute
#ex to pull up a table of all cells in batch #271, enter address
# "http://127.0.0.1:5000/celllist/batch=271" (or whatever your default server address:port is)

#basic landing page w/ dropboxe for user batch choice
@app.route("/index")
def index_view():
    data = read_table('NarfBatches')
    batchlist = data['batch'].tolist()
    batchlist = list(set(batchlist))
    return render_template('index.html', batchlist = batchlist)


#basic functions for querying based on batch, model etc

#pull cells with matching batchid
@app.route("/celllist/batch=<batchid>")
def show_batch(batchid):
    data = query_batch(batchid)
    data.set_index(['id'], inplace=True)
    data.index.name=None
    return render_template('base.html', table=data.to_html(classes='Table'))

#pull cells with matching modelname
@app.route("/celllist/modelname=<modelname>")
def show_model(modelname):
    data = query_model(modelname)
    data.set_index(['id'], inplace=True)
    data.index.name=None
    return render_template('base.html', table=data.to_html(classes='Table'))

#pull cells with matching batchid and modelname (in that order)
@app.route("/celllist/batch=<batchid>/modelname=<modelname>")
def show_batch_plus_model(batchid, modelname):
    data = query_model_from_batch(batchid, modelname)
    data.set_index(['id'], inplace=True)
    data.index.name=None
    return render_template('base.html', table=data.to_html(classes='Table'))



#basic table pulls below, no filtering
@app.route("/NarfResults")
def show_narfresults():
    data = read_table('NarfResults')
    data.set_index(['id'], inplace=True)
    data.index.name=None
    return render_template('base.html', table=data.to_html(classes='Table'))

@app.route("/NarfBatches")
def show_narfbatches():
    data = read_table('NarfBatches')
    data.set_index(['id'], inplace=True)
    data.index.name=None
    
    return render_template('base.html',table=data.to_html(classes='Table'))
    
@app.route("/gSingleCell")
def show_gsinglecell():
    data = read_table('gSingleCell')
    
    return render_template('base.html',table=data.to_html(classes='Table'))
    
if __name__ == "__main__":
    app.run(debug=False)
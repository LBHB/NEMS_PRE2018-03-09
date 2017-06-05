from flask import render_template, jsonify, request
from nems_analysis import app, Session, NarfAnalysis, NarfBatches, NarfResults
from nems_analysis.ModelFinder import ModelFinder
#moved views for these to separate plot_functions.views
#from nems_analysis.PlotGenerator import Scatter_Plot, Bar_Plot, Pareto_Plot
import pandas.io.sql as psql
from sqlalchemy.orm import Query
from sqlalchemy import desc, asc

# testing navigation to modelpane template
@app.route('/modelpane')
def modelpane_view():
    return render_template('/modelpane/modelpane.html')

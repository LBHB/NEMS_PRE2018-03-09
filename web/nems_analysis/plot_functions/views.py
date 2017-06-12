""" additional views for calling plot functions """

from flask import render_template, jsonify, request, Response
from nems_analysis import app, Session, NarfResults
from plot_functions.PlotGenerator import Scatter_Plot, Bar_Plot, Pareto_Plot
import pandas.io.sql as psql
import numpy as np

@app.route('/scatter_plot', methods=['GET','POST'])
def scatter_plot():
    session = Session()
    
    bSelected = request.form.get('batch')[:3]
    mSelected = request.form.getlist('modelnames')
    cSelected = request.form.getlist('celllist')
    measure = request.form['measure']
    #onlyFair = request.form.get('onlyFair')
    onlyFair = True
    #includeOutliers = request.form.get('includeOutliers')
    includeOutliers = True
    
    useSNRorIso = (request.form.get('plotOption[]'),request.form.get('plotOpVal'))
    
    results = psql.read_sql_query(session.query(getattr(NarfResults,measure),NarfResults.cellid,\
              NarfResults.modelname).filter(NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).statement,session.bind)
              
    if results.size == 0:
        return Response("empty plot")
    
    # TODO: filter results based on useSNRorIso before passing data to plot generator
    # note: doing this here instead of in plot generator since it requires db access
    #       make a list of cellids that fail snr/iso criteria
    #       then remove all rows of results where cellid is in that list
    
    plot = Scatter_Plot(data=results,fair=onlyFair,outliers=includeOutliers,\
                        measure=measure)
    plot.generate_plot()
    
    session.close()
    
    return render_template("/plot/plot.html", script=plot.script, div=plot.div)


@app.route('/bar_plot',methods=['GET','POST'])
def bar_plot():
    session = Session()
    
    bSelected = request.form.get('batch')[:3]
    mSelected = request.form.getlist('modelnames')
    cSelected = request.form.getlist('celllist')
    measure = request.form['measure']
    #onlyFair = request.form.get('onlyFair')
    onlyFair = True
    #includeOutliers = request.form.get('includeOutliers')
    includeOutliers = True
    
    useSNRorIso = (request.form.get('plotOption[]'),request.form.get('plotOpVal'))
    
    results = psql.read_sql_query(session.query(getattr(NarfResults,measure),NarfResults.cellid,\
              NarfResults.modelname).filter(NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).statement,session.bind)
              
    if results.size == 0:
        return Response("empty plot")
    
    plot = Bar_Plot(data=results,fair=onlyFair,outliers=includeOutliers,\
                        measure=measure)
    plot.generate_plot()
    
    session.close()
    
    return render_template("/plot/plot.html",script=plot.script,div=plot.div)


@app.route('/pareto_plot',methods=['GET','POST'])
def pareto_plot():
    session = Session()
    
    bSelected = request.form.get('batch')[:3]
    mSelected = request.form.getlist('modelnames')
    cSelected = request.form.getlist('celllist')
    measure = request.form['measure']
    #onlyFair = request.form.get('onlyFair')
    onlyFair = True
    #includeOutliers = request.form.get('includeOutliers')
    includeOutliers = True
    
    useSNRorIso = (request.form.get('plotOption[]'),request.form.get('plotOpVal'))
    
    results = psql.read_sql_query(session.query(getattr(NarfResults,measure),NarfResults.cellid,\
              NarfResults.modelname).filter(NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).statement,session.bind)
              
    if results.size == 0:
        return Response("empty plot")
    
    plot = Pareto_Plot(data=results,fair=onlyFair,outliers=includeOutliers,\
                        measure=measure)
    plot.generate_plot()
    
    session.close()
    
    return render_template("/plot/plot.html",script=plot.script,div=plot.div)


@app.route('/plot_strf')
def plot_strf():
    session = Session()
    # will need to get selections from results table using ajax, instead of
    # using a form submission like the above plots.
    session.close()
    return Response('STRF view function placeholder')
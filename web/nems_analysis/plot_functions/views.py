""" additional views for calling plot functions """

from flask import render_template, jsonify, request, Response
from nems_analysis import app, Session, NarfResults
from plot_functions.PlotGenerator import Scatter_Plot, Bar_Plot, Pareto_Plot
import pandas.io.sql as psql

@app.route('/scatter_plot', methods=['GET','POST'])
def scatter_plot():
    session = Session()
    
    bSelected = request.form.get('batch')[:3]
    mSelected = request.form.getlist('modelnames')
    cSelected = request.form.getlist('celllist')
    measure = request.form['measure']
    
    results = psql.read_sql_query(session.query(getattr(NarfResults,measure),NarfResults.cellid,\
              NarfResults.modelname).filter(NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).statement,session.bind)
              
    plot = Scatter_Plot(data=results,celllist=cSelected,modelnames=mSelected,\
                        measure=measure, batch=bSelected)
    plot.generate_plot()
    
    session.close()
    
    return render_template("/plot/plot.html", script=plot.script, div=plot.div)


@app.route('/bar_plot',methods=['GET','POST'])
def bar_plot():
    session = Session()
    # TODO: this is exactly the same as scatter_plot other than the function call
    # to Bar_Plot instead of Scatter_Plot - should these be combined into a
    # single function or left separate for clarity?
    
    bSelected = request.form.get('batch')[:3]
    mSelected = request.form.getlist('modelnames')
    cSelected = request.form.getlist('celllist')
    measure = request.form['measure']
    
    results = psql.read_sql_query(session.query(getattr(NarfResults,measure),NarfResults.cellid,\
              NarfResults.modelname).filter(NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).statement,session.bind)
        
    plot = Bar_Plot(data=results,celllist=cSelected,modelnames=mSelected,\
                    measure=measure,batch=bSelected)
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
    
    results = psql.read_sql_query(session.query(getattr(NarfResults,measure),\
              NarfResults.cellid,NarfResults.n_parms,\
              NarfResults.modelname).filter(NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).statement,session.bind)
        
    plot = Pareto_Plot(data=results,celllist=cSelected,modelnames=mSelected,\
                    measure=measure,batch=bSelected)
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
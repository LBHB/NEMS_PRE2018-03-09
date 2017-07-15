"""View functions for handling Submit Plot button.

Query database with batch, cell and model selection.
Filter out cells that don't meet minimum SNR/Iso/SNRi criteria.
Pass the returned NarfResults dataframe - along with measure, onlyFair and
includeOutliers criterio - to a PlotGenerator subclass based on the selected
plot type.
Use that PlotGenerator's generate_plot() method to generate <script> and <div>
components to pass back to the JS ajax function, which will insert them into
the display area in the browser.

"""

import pandas.io.sql as psql

from flask import render_template, jsonify, request, Response

from nems_analysis import app, Session, NarfResults, NarfBatches
import plot_functions.PlotGenerator as pg

        
@app.route('/generate_plot_html')
def generate_plot_html():

    session = Session()
    
    plotType = request.args.get('plotType')
    bSelected = request.args.get('bSelected')[:3]
    mSelected = request.args.getlist('mSelected[]')
    cSelected = request.args.getlist('cSelected[]')
    measure = request.args.get('measure')
    onlyFair = request.args.get('onlyFair')
    if int(onlyFair):
        onlyFair = True
    else:
        onlyFair = False
    includeOutliers = request.args.get('includeOutliers')
    if int(includeOutliers):
        includeOutliers = True
    else:
        includeOutliers = False
    
    # TODO: Re-do this to include any new criteria dynamically instead of
    #       hard-coding snr/iso/snri.
    filterCriteria = {
            'snr' : float(request.args.get('snr')),
            'iso' : float(request.args.get('iso')),
            'snri' : float(request.args.get('snri')),
            }
    # TODO: Looks like this is what NARF does, but not 100% sure.
    # Always exclude negative values
    for key in filterCriteria:
        if filterCriteria[key] < 0:
            filterCriteria[key] = 0
            
    removalCount = 0
    for cellid in cSelected:
        dbCriteria = (
                session.query(NarfBatches)
                .filter(NarfBatches.batch == bSelected)
                .filter(NarfBatches.cellid == cellid)
                .all()
                )
        if dbCriteria:
            if len(dbCriteria) > 1:
                print(
                    "Multiple results found for cellid: %s in batch: %s"
                    %(cellid, bSelected)
                    )
            min_snr = min(dbCriteria[0].est_snr, dbCriteria[0].val_snr)
            min_isolation = dbCriteria[0].min_isolation
            min_snr_index = dbCriteria[0].min_snr_index
            
            a = (filterCriteria['snr'] > min_snr)
            b = (filterCriteria['iso'] > min_isolation)
            c = (filterCriteria['snri'] > min_snr_index)
            
            if a or b or c:
                filterReason = ""
                if a:
                    filterReason += (
                            "min snr: %s -- was less than criteria: %s\n"
                            %(min_snr, filterCriteria['snr'])
                            )
                if b:
                    filterReason += (
                            "min iso: %s -- was less than criteria: %s\n"
                            %(min_isolation, filterCriteria['iso'])
                            )
                if c:
                    filterReason += (
                            "min snr index: %s -- was less than criteria: %s\n"
                            %(min_snr_index, filterCriteria['snri'])
                            )
                #print(
                #    "Removing cellid: %s,\n"
                #    "because: %s"
                #    %(cellid, filterReason)
                #    )
                cSelected.remove(cellid)
                removalCount += 1
        else:
            print(
                "No entry in NarfBatches for cellid: %s in batch: %s"
                %(cellid, bSelected)
                )
            cSelected.remove(cellid)
            removalCount += 1
    print("Number of cells filtered due to snr/iso criteria: %d"%removalCount)
        
    results = psql.read_sql_query(
            session.query(NarfResults)
            .filter(NarfResults.batch == bSelected)
            .filter(NarfResults.cellid.in_(cSelected))
            .filter(NarfResults.modelname.in_(mSelected))
            .statement,
            session.bind
            )
    
    Plot_Class = getattr(pg, plotType)
    plot = Plot_Class(
            data=results, measure=measure, fair=onlyFair, 
            outliers=includeOutliers,
            )
    if plot.emptycheck:
        print('Plot checked empty after forming data array')
        return jsonify(script='Empty',div='Plot')
    else:
        plot.generate_plot()
        
    session.close()
    
    if hasattr(plot, 'script') and hasattr(plot, 'div'):
        return jsonify(script=plot.script, div=plot.div)
    elif hasattr(plot, 'html'):
        return jsonify(html=plot.html)
    else:
        return jsonify(script="Couldn't find anything ", div="to return")
    
    


######      DEPRECATED BELOW        ######

@app.route('/scatter_plot', methods=['GET','POST'])
def scatter_plot():
    """Pass user selections to a Scatter_Plot object, then display the results
    of generate_plot.
    
    """
    
    session = Session()
    # Call script to get Plot Generator arguments from user selections.
    args = load_plot_args(request, session)
    if args['data'].size == 0:
        return Response("empty plot")
    
    plot = Scatter_Plot(**args)
    # Check plot data to see if everything got filtered out by data formatter.
    if plot.emptycheck:
        return Response("empty plot")
    else:
        plot.generate_plot()
    
    session.close()
    
    return render_template("/plot/plot.html", script=plot.script, div=plot.div)


@app.route('/bar_plot',methods=['GET','POST'])
def bar_plot():
    """Pass user selections to a Bar_Plot object, then display the results
    of generate_plot.
    
    """
    
    session = Session()
    # Call script to get Plot Generator arguments from user selections.
    args = load_plot_args(request,session)
    if args['data'].size == 0:
        return Response("empty plot")
    
    plot = Bar_Plot(**args)
    # Check plot data to see if everything got filtered out by data formatter.
    if plot.emptycheck:
        return Response("empty plot")
    else:
        plot.generate_plot()
    
    session.close()
    
    return render_template("/plot/plot.html",script=plot.script,div=plot.div)


@app.route('/pareto_plot',methods=['GET','POST'])
def pareto_plot():
    """Pass user selections to a Pareto_Plot object, then display the
    results of generate_plot.
    
    """
    
    session = Session()
    
    # Call script to get Plot Generator arguments from user selections.
    args = load_plot_args(request,session)
    if args['data'].size == 0:
        return Response("empty plot")
    
    plot = Pareto_Plot(**args)
    # Check plot data to see if everything was filtered out by data formatter.
    if plot.emptycheck:
        return Response("empty plot")
    else:
        plot.generate_plot()
    
    session.close()
    
    return render_template("/plot/plot.html",script=plot.script,div=plot.div)


@app.route('/plot_strf')
def plot_strf():
    """Not yet implemented."""
    
    session = Session()
    # will need to get selections from results table using ajax, instead of
    # using a form submission like the above plots.
    session.close()
    return Response('STRF view function placeholder')


def load_plot_args(request, session):
    """Combines user selections and database entries into a dict of arguments.
    
    Queries database based on user selections for batch, cell and modelname and
    packages the results into a Pandas DataFrame. The DataFrame, along with
    the performance measure, fair and outliers options from the nems_analysis
    web interface are then packaged into a dict structure to match the
    argument requirements of the Plot_Generator base class.
    Since all Plot_Generator objects use the same required arguments, this
    eliminates the need to repeat the selection and querying code for every
    view function.
    
    Arguments:
    ----------
    request : flask request context
        Current request context generated by flask. See flask documentation.
    session : sqlalchemy database session
        An open transaction with the database. See sqlalchemy documentation.
        
    Returns:
    --------
    {} : dict-like
        A dictionary specifying the arguments that should be passed to a
        Plot_Generator object.
    
    Note:
    -----
    This adds no additional functionality, it is only used to simplify
    the code for the above view functions. If desired, it can be copy-pasted
    back into the body of each view function instead, with few changes.
    
    """
    
    bSelected = request.form.get('batch')[:3]
    mSelected = request.form.getlist('modelnames[]')
    cSelected = request.form.getlist('celllist[]')
    measure = request.form['measure']
    onlyFair = request.form.get('onlyFair')
    if onlyFair == "fair":
        onlyFair = True
    else:
        onlyFair = False
    includeOutliers = request.form.get('includeOutliers')
    if includeOutliers == "outliers":
        includeOutliers = True
    else:
        includeOutliers = False
    
    #useSNRorIso = (request.form.get('plotOption[]'),request.form.get('plotOpVal'))
    
    # TODO: filter results based on useSNRorIso before passing data to plot generator
    # note: doing this here instead of in plot generator since it requires db access
    #       make a list of cellids that fail snr/iso criteria
    #       then remove all rows of results where cellid is in that list
    
    results = psql.read_sql_query(session.query(NarfResults).filter\
              (NarfResults.batch == bSelected).filter\
              (NarfResults.cellid.in_(cSelected)).filter\
              (NarfResults.modelname.in_(mSelected)).statement,session.bind)
    
    return {
        'data':results,'measure':measure,'fair':onlyFair,
        'outliers':includeOutliers,
        }
    
    
    
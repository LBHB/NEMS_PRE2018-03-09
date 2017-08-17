"""Miscellaneous view functions.

Contents so far:
    status_report
    
"""

import time
import itertools

import pandas.io.sql as psql
import pandas as pd
import numpy as np
from flask import request, render_template, jsonify

from nems.web.nems_analysis import app
from nems.db import Session, NarfResults, cluster_tQueue, cluster_Session
from nems.web.plot_functions.reports import Performance_Report, Fit_Report
from nems.speed_test import Timer

@app.route('/batch_performance', methods=['GET', 'POST'])
def batch_performance():
    session = Session()
    
    cSelected = request.form['cSelected']
    bSelected = request.form['bSelected'][:3]
    mSelected = request.form['mSelected']
    findAll = request.form['findAll']
    
    cSelected = cSelected.split(',')
    mSelected = mSelected.split(',')
    
    if int(findAll):
        results = psql.read_sql_query(
                session.query(
                        NarfResults.cellid, NarfResults.modelname,
                        NarfResults.r_test
                        )
                .filter(NarfResults.batch == bSelected)
                .statement,
                session.bind
                )
    else:
        results = psql.read_sql_query(
                session.query(
                        NarfResults.cellid, NarfResults.modelname,
                        NarfResults.r_test
                        )
                .filter(NarfResults.batch == bSelected)
                .filter(NarfResults.cellid.in_(cSelected))
                .filter(NarfResults.modelname.in_(mSelected))
                .statement,
                session.bind
                )
                
    
    report = Performance_Report(results, bSelected)
    report.generate_plot()
        
    session.close()
    return render_template(
            'batch_performance.html', script=report.script, div=report.div
            )
    
@app.route('/fit_report')
def fit_report():
    session = Session()
    cluster_session = cluster_Session()
    
    cSelected = request.args.getlist('cSelected[]')
    bSelected = request.args.get('bSelected')[:3]
    mSelected = request.args.getlist('mSelected[]')
    
    multi_index = pd.MultiIndex.from_product(
            [mSelected, cSelected], names=['modelname', 'cellid']
            )
    status = pd.DataFrame(index=multi_index, columns=['yn'])
    
    # TODO: the nested queries are causing the majority of the sluggishness,
    #       especially the cluster_session queries since they have to route
    #       through bhangra. need to figure out a way to do this with 1 query.
    tuples = list(itertools.product(cSelected, [bSelected], mSelected))
    notes = ['{0}/{1}/{2}'.format(t[0],t[1],t[2]) for t in tuples]
    
    qdata = psql.read_sql_query(
            cluster_session.query(cluster_tQueue)
            .filter(cluster_tQueue.note.in_(notes))
            .statement,
            cluster_session.bind,
            )

    results = psql.read_sql_query(
            session.query(
                    NarfResults.cellid, NarfResults.batch,
                    NarfResults.modelname,
                    )
            .filter(NarfResults.batch == bSelected)
            .filter(NarfResults.cellid.in_(cSelected))
            .filter(NarfResults.modelname.in_(mSelected))
            .statement,
            session.bind
            )

    for i, t in enumerate(tuples):
        yn = 3 # missing
        try:
            complete = qdata.loc[qdata['note'] == notes[i], 'complete'].iloc[0]
            if complete < 0:
                yn = 4 # in progress
            elif complete == 0:
                yn = 5 # not started
            elif complete == 1:
                yn = 6# finished
            elif complete == 2:
                yn = 0 # dead entry
            else:
                pass # unknown value, so leave as missing?
        except:
            try:
                result = results.loc[
                        (results['cellid'] == t[0])
                        & (results['batch'] == int(t[1]))
                        & (results['modelname'] == t[2]),
                        'cellid'
                        ].iloc[0]
                yn = 6
            except:
                pass
        status['yn'].loc[t[2],t[0]] = yn
    
    status.reset_index(inplace=True)
    status = status.pivot(index='cellid', columns='modelname', values='yn')
    status = status[status.columns].astype(int)
    report = Fit_Report(status)
    report.generate_plot()
    
    session.close()
    cluster_session.close()
    return jsonify(
            html=(report.html)
            )
                    
    
    # leaving this here until new implementation (above) has been tested some
    # more, but shouldn't need this as long as it continues to work.
    """
    for model in mSelected:
        for cell in cSelected:
            yn = 3 # missing
            note = "%s/%s/%s"%(cell, bSelected, model)
            #start = time.time()
            qdata = (
                    cluster_session.query(cluster_tQueue)
                    .filter(cluster_tQueue.note == note)
                    .first()
                    )
            #end = time.time()
            #print('elapsed time for qdata query: %s s'%(end-start))
            if qdata:
                # The values are changed around here to make the color spread
                # of the heatmap follow a logical progression.
                # They should not be confused with the actual values of the
                # 'complete' column in tQueue.
                # -Jacob, 8/17/2017
                if qdata.complete < 0:
                    yn = 2 # in progress
                elif qdata.complete == 0:
                    yn = 1 # not yet started
                elif qdata.complete == 1:
                    yn = 0 # finished
                elif qdata.complete == 2:
                    yn = 6 # dead entry
                else:
                    pass # unknown value, so leave as missing?
            #start = time.time()
            else:
                result = (
                        session.query(NarfResults)
                        .filter(NarfResults.batch == bSelected)
                        .filter(NarfResults.cellid == cell)
                        .filter(NarfResults.modelname == model)
                        .first()
                        )
                #end = time.time()
                #print('elapsed time for results query: %s s'%(end-start))
                if result:
                    yn = 0 # finished
                else:
                    pass
                
            status['yn'].loc[model,cell] = yn
    """
                    
    
    
    
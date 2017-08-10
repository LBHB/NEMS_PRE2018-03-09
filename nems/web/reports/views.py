"""Miscellaneous view functions.

Contents so far:
    status_report
    
"""

import pandas.io.sql as psql
import pandas as pd
import numpy as np
from flask import request, render_template

from nems.web.nems_analysis import app
from nems.db import Session, NarfResults, cluster_tQueue, cluster_Session
from nems.web.plot_functions.Status_Report import Status_Report

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
                
    report = Status_Report(results, bSelected)
    report.generate_plot()
        
    session.close()
    return render_template(
            'batch_performance.html', script=report.script, div=report.div
            )
    
@app.route('/fit_report', methods=['GET', 'POST'])
def fit_report():
    session = Session()
    cluster_session = cluster_Session()
    
    cSelected = request.form['cSelected']
    bSelected = request.form['bSelected'][:3]
    mSelected = request.form['mSelected']
    
    cSelected = cSelected.split(',')
    mSelected = mSelected.split(',')
    
    # TODO: build dataframe directly instead of building dict then reading
    #       into pandas
    
    # TODO: what kind of plot for this?
    
    status = {}
    for i, model in enumerate(mSelected):
        status.update({model:{}})
        for cell in cSelected:
            yn = np.nan
            note = "%s/%s/%s"%(cell, bSelected, model)
            
            qdata = (
                    cluster_session.query(cluster_tQueue)
                    .filter(cluster_tQueue.note == note)
                    .first()
                    )
            if qdata:
                yn = qdata.complete
            else:
                result = (
                        session.query(NarfResults)
                        .filter(NarfResults.batch == bSelected)
                        .filter(NarfResults.cellid.in_(cSelected))
                        .filter(NarfResults.modelname.in_(mSelected))
                        .first()
                        )
                if result:
                    yn = 'X'
                else:
                    pass
            status[model].update({cell:yn})

    table = pd.DataFrame(status).T.to_html()

    session.close()
    cluster_session.close()
    return render_template(
            'fit_report.html', html=table,
            )
                    
                    
    
    
    
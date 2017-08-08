"""Miscellaneous view functions.

Contents so far:
    status_report
    
"""

import pandas.io.sql as psql
from flask import request, jsonify, render_template

from nems.web.nems_analysis import app
from nems.db import Session, NarfResults
from nems.web.plot_functions.Status_Report import Status_Report

@app.route('/status_report', methods=['GET', 'POST'])
def status_report():
    session = Session()
    
    bSelected = request.form['bSelected'][:3]
    
    results = psql.read_sql_query(
            session.query(
                    NarfResults.cellid, NarfResults.modelname, NarfResults.r_test
                    )
            .filter(NarfResults.batch == bSelected)
            .statement,
            session.bind
            )
    
    report = Status_Report(results, bSelected)
    report.generate_plot()
    
    return render_template(
            'status_report.html', script=report.script, div=report.div
            )
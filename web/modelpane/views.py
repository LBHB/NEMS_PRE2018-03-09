"""Placeholder views function file for modelpane.

Only used for testing the template right now.

"""

import copy

from flask import request, render_template, Session, Response, jsonify
import matplotlib.pyplot as plt, mpld3

import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.nems_main as nems

from nems_analysis import app

FIGSIZE = (12,4) # width, height for matplotlib figures
mp_stack = None
          
@app.route('/modelpane', methods=['GET','POST'])
def modelpane_view():
    """Launches Modelpane window to inspect the model fit to the selected
    cell and model."""

    # reset stack if modelpane is restarted
    global mp_stack
    mp_stack = None

    
    bSelected = request.form.get('batch')[:3]
    cSelected = request.form.get('cellid')
    mSelected = request.form.get('modelname')

    try:
        mp_stack = nems.load_single_model(
                cellid=cSelected, batch=bSelected, modelname=mSelected,
                )
    except:
        return Response(
                "Model has not been fitted yet, or its fit file",
                "is not in local storage."
                )
        
    stackmods = mp_stack.modules[1:]
    plots = []
    for m in stackmods:
        p = plt.figure(figsize=FIGSIZE)
        m.do_plot(m)
        html = mpld3.fig_to_html(p)
        plots.append(html)
        plt.close(p)
    # make double sure that all figures close after loop
    # to avoid excess memory usage.
    plt.close("all")
    
    plot_fns = copy.deepcopy([m.plot_fns for m in stackmods])
    for i, pf in enumerate(plot_fns):
        for j, f in enumerate(pf):
            newf = printable_plot_name(f)
            plot_fns[i][j] = newf
            
    return render_template(
            "/modelpane/modelpane.html", 
            modules=[m.name for m in stackmods],
            plots=plots,
            title="Cellid: %s --- Model: %s"%(cSelected,mSelected),
            fields=[m.user_editable_fields for m in stackmods],
            plottypes=plot_fns,
           )
   

@app.route('/update_modelpane_plot')
def update_modelpane_plot():
    #"""Placeholder functions. Update some parameter/attribute of a module?"""
    
    global mp_stack
    modAffected = request.args.get('modAffected')
    plotType = request.args.get('plotType')
    if not modAffected:
        return jsonify(html="<p>Affected module is None<p>")
    
    try:
        i = [mod.name for mod in mp_stack.modules].index(modAffected)
    except Exception as e:
        print(e)
        return Response('')
    
    try:
        m = mp_stack.modules[i]
    except Exception as e:
        print(e)
        print("index was: " + str(i))
        return Response('')
    
    p = plt.figure(figsize=FIGSIZE)
    plot_fn = getattr(nu, plotType)
    plot_fn(m)
    html = mpld3.fig_to_html(p)
    
    return jsonify(html=html)

def printable_plot_name(plot_fn):
    p = str(plot_fn).replace('<function ', '')
    i = p.find(' at')
    if i > 0:
        p = p[:i]
    return p
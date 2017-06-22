"""Placeholder views function file for modelpane.

Only used for testing the template right now.

"""

from flask import request, render_template, Session, Response
import matplotlib.pyplot as plt, mpld3

import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.nems_main as nems

from nems_analysis import app

@app.route('/modelpane', methods=['GET','POST'])
def modelpane_view():
    """Launches Modelpane window to inspect the model fit to the selected
    cell and model."""

    bSelected = request.form.get('batch')[:3]
    cSelected = request.form.get('cellid')
    mSelected = request.form.get('modelname')

    stack = nm.nems_stack()
    try:
        stack = nems.load_single_model(
                cellid=cSelected, batch=bSelected, modelname=mSelected,
                )
    except:
        return Response(
                "Model has not been fitted yet, or its fit file",
                "is not in local storage."
                )

    plots = []
    for m in stack.modules[1:]:
        p = plt.figure(figsize=(12,4))
        m.do_plot(m)
        html = mpld3.fig_to_html(p)
        plots.append(html)
        plt.close(p)
    # make double sure that all figures close after loop
    # to avoid excess memory usage.
    plt.close("all")
    
    plot_fns = [m.plot_fns for m in stack.modules[1:]]    
    for pf in plot_fns:
        for f in pf:
            f = printable_plot_name(f)
    
    return render_template(
            "/modelpane/modelpane.html", 
            modules=[m.name for m in stack.modules[1:]],
            plots=plots,
            title="Cellid: %s --- Model: %s"%(cSelected,mSelected),
            fields=[m.user_editable_fields for m in stack.modules[1:]],
            plottypes=plot_fns
           )
   

#@app.route('/update_modelpane')
#def update_modelpane():
    #"""Placeholder functions. Update some parameter/attribute of a module?"""
    
    # Triggered if a selection is made in modelpane. Get the new specification
    # and add it to/overwrite it in the appropriate module.
    #modAffected = request.args.get('modAffected')
    #modSpec = request.args.get('specChanged')
    #modValue = request.args.get('newSpecValue')
    
    #modObject = Flask.g['modulesDict'][modAffected]
    #modObject.setattr(modSpec, modValue)
    #plot = modObject.make_new_plot_with_changed_value
    
    #return jsonify(plot=plot)

def printable_plot_name(plot_fn):
    p = plot_fn.replace('<function plot_', '')
    i = p.find(' at')
    p = p[:i]
    return p
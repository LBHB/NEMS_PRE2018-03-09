"""Placeholder views function file for modelpane.

Only used for testing the template right now.

"""

import copy

from flask import (
        request, render_template, Session, Response, jsonify, redirect, 
        url_for,
        )
import matplotlib.pyplot as plt, mpld3

import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_keywords as nk
import lib.nems_utils as nu
import lib.nems_main as nems

from nems_analysis import app

FIGSIZE = (12,4) # width, height for matplotlib figures
mp_stack = None
          
@app.route('/modelpane_view', methods=['GET','POST'])
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
        
    all_mods = [cls.name for cls in nm.nems_module.__subclasses__()]
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
    
    # Need to use copy to avoid overriding modules' attributes
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
            all_mods=all_mods,
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


@app.route('/refresh_modelpane')
def refresh_modelpane():
    """Copy of modelpane_view code after loading model.
    Returns dict of arguments for re-rendering the modelpane template.
    
    """
    global mp_stack
    cell = mp_stack.meta['cellid']
    model = mp_stack.meta['modelname']
    
    all_mods = [cls.name for cls in nm.nems_module.__subclasses__()]
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
    
    # Need to use copy to avoid overriding modules' attributes
    plot_fns = copy.deepcopy([m.plot_fns for m in stackmods])
    for i, pf in enumerate(plot_fns):
        for j, f in enumerate(pf):
            newf = printable_plot_name(f)
            plot_fns[i][j] = newf

    return render_template(
            'modelpane/modelpane.html',
            modules=[m.name for m in stackmods],
            plots=plots,
            title="Cellid: %s --- Model: %s"%(cell, model),
            fields=[m.user_editable_fields for m in stackmods],
            plottypes=plot_fns,
            all_mods=all_mods,
            )

@app.route('/append_module', methods=['GET','POST'])
def append_module():

    global mp_stack
    
    module_name = request.form.get('a_module')
    
    try:
        m = getattr(nm, module_name)
    except Exception as e:
        print(e)
        m = None
    try:    
        mp_stack.append(m)
    except Exception as e:
        print("Exception inside nems_modules > stack > append()")
        print(e)
    
    return redirect(url_for('refresh_modelpane'))

@app.route('/insert_module', methods=['GET','POST'])
def insert_module():
    
    #TODO: Not currently working - see lib.nems_modules > nems_stack > insert()
    
    global mp_stack
    
    module_name = request.form.get('a_module')
    idx = request.form.get('idx')
    if not idx:
        print("No insertion index selected")
        return redirect(url_for('refresh_modelpane'))
    idx = int(idx)
    
    try:
        m = getattr(nm, module_name)
    except Exception as e:
        print(e)
        m = None
    try:
        mp_stack.insert(mod=m, idx=idx)
    except Exception as e:
        print("Exception inside nem_modules > stack > insert()")
        print(e)

    return redirect(url_for('refresh_modelpane'))

@app.route('/remove_module', methods=['GET','POST'])
def remove_module():
    
    global mp_stack
    
    module_name = request.form.get('r_module')
    all_idx = request.form.get('all_idx')
    if not all_idx:
        all_idx = False
    else:
        all_idx = True
    
    try:
        m = getattr(nm, module_name)
    except Exception as e:
        print(e)
        m = None
    try:
        mp_stack.remove(mod=m, all_idx=all_idx)
    except Exception as e:
        print("Exception inside nems_modules > stack > remove()")
        print(e)

    # refresh modelpane view
    return redirect(url_for('refresh_modelpane'))


def printable_plot_name(plot_fn):
    p = str(plot_fn).replace('<function ', '')
    i = p.find(' at')
    if i > 0:
        p = p[:i]
    return p

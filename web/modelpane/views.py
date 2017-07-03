"""Placeholder views function file for modelpane.

Only used for testing the template right now.

"""

import copy
import inspect

from flask import (
        request, render_template, Session, Response, jsonify, redirect, 
        url_for,
        )
import matplotlib.pyplot as plt, mpld3
import numpy as np

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
                "Model has not been fitted yet, or its fit file "
                "is not in local storage."
                )
        
    all_mods = [cls.name for cls in nm.nems_module.__subclasses__()]
    # remove any modules that shouldn't show up as an option in modelpane
    all_mods.remove('load_mat')
    
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
    
    fields = [m.user_editable_fields for m in stackmods]
    values = [
            [getattr(m, field) for field in fields[i]]
            for i, m in enumerate(stackmods)
            ]
    types = [
            [str(type(getattr(m, field)))
            .replace("<class '","").replace("'>","")
            for field in fields[i]]
            for i, m in enumerate(stackmods)
            ]
    #for i, itr in enumerate(types):
    #    for j, t in enumerate(itr):
    #        t = t.replace("<class '","")
    #        t = t.replace("'>","")
    #        types[i][j] = t
    fields_values_types = []
    for i, itr in enumerate(fields):
        fields_values_types.append(zip(fields[i], values[i], types[i]))
            
    
    #keywords = [
    #        [m.name] + vars(nk)[m.name]
    #        for m in stackmods
    #        if m.name in vars(nk)
    #        ]
    
    # TODO: how to calculate stim and data idx range from stack.data?
    stim_max = 0
    data_max = 1
    
    return render_template(
            "/modelpane/modelpane.html", 
            modules=[m.name for m in stackmods],
            plots=plots,
            title="Cellid: %s --- Model: %s"%(cSelected,mSelected),
            fields_values_types=fields_values_types,
            #keywords=keywords,
            plottypes=plot_fns,
            all_mods=all_mods,
            plot_stimidx=mp_stack.plot_stimidx,
            plot_dataidx=mp_stack.plot_dataidx,
            plot_stimidx_max=stim_max,
            plot_dataidx_max=data_max,
           )
   

@app.route('/refresh_modelpane')
def refresh_modelpane():
    """Copy of modelpane_view code after loading model.
    Returns dict of arguments for re-rendering the modelpane template.
    
    """
    
    # TODO: Better way to do this instead of copy-pasting updates from main
    #        view function? Should convert to ajax eventually too instead of
    #        full refresh.
    global mp_stack
    cell = mp_stack.meta['cellid']
    model = mp_stack.meta['modelname']
    
    all_mods = [cls.name for cls in nm.nems_module.__subclasses__()]
    # remove any modules that shouldn't show up as an option in modelpane
    all_mods.remove('load_mat')
    
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
    
    fields = [m.user_editable_fields for m in stackmods]
    values = [
            [getattr(m, field) for field in fields[i]]
            for i, m in enumerate(stackmods)
            ]
    types = [
            [str(type(getattr(m, field)))
            .replace("<class '","").replace("'>","")
            for field in fields[i]]
            for i, m in enumerate(stackmods)
            ]
    #for i, itr in enumerate(types):
    #    for j, t in enumerate(itr):
    #        t = t.replace("<class '","")
    #        t = t.replace("'>","")
    #        types[i][j] = t
    fields_values_types = []
    for i, itr in enumerate(fields):
        fields_values_types.append(zip(fields[i], values[i], types[i]))
            
    
    #keywords = [
    #        [m.name] + vars(nk)[m.name]
    #        for m in stackmods
    #        if m.name in vars(nk)
    #        ]
    
    # TODO: how to calculate stim and data idx range from stack.data?
    stim_max = 0
    data_max = 1
    
    return render_template(
            "/modelpane/modelpane.html", 
            modules=[m.name for m in stackmods],
            plots=plots,
            title="Cellid: %s --- Model: %s"%(cell ,model),
            fields_values_types=fields_values_types,
            #keywords=keywords,
            plottypes=plot_fns,
            all_mods=all_mods,
            plot_stimidx=mp_stack.plot_stimidx,
            plot_dataidx=mp_stack.plot_dataidx,
            plot_stimidx_max=stim_max,
            plot_dataidx_max=data_max,
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
                 

@app.route('/update_idx')
def update_idx():
    
    global mp_stack
    plot_stimidx = request.args.get('plot_stimidx')
    plot_dataidx = request.args.get('plot_dataidx')
    
    mp_stack.plot_stimidx = int(plot_stimidx)
    mp_stack.plot_dataidx = int(plot_dataidx)
    
    stackmods = mp_stack.modules[1:]
    plots = []
    for m in stackmods:
        p = plt.figure(figsize=FIGSIZE)
        m.do_plot(m)
        html = mpld3.fig_to_html(p)
        plots.append(html)
        plt.close(p)
    
    return jsonify(plots=plots)
    

@app.route('/update_module')
def update_module():
    
    global mp_stack
    fields = request.args.getlist('fields[]')
    values = request.args.getlist('values[]')
    types = request.args.getlist('types[]')
    if (not fields) or (not values) or (not types):
        raise ValueError("No fields and/or values and/or types came through")
    fields_values = zip(fields, values, types)
    
    modAffected = request.args.get('modAffected')
    modIdx = nu.find_modules(mp_stack, modAffected)
    if modIdx:
        modIdx = modIdx[0]
    else:
        print("Module could not be found in stack.")
        return jsonify(success=False)

    for f, v, t in fields_values:
        #TODO: figure out a good way to set type dynamically instead of trying
        #      to think of every possible data type."
        if not hasattr(mp_stack.modules[modIdx], f):
            raise AttributeError("Couldn't find attribute for module.")
        if t == "NoneType":
            v = None
        elif t == "str":
            pass
        elif t == "int":
            v = int(v)
        elif t == "float":
            v = float(v)
        elif t == "numpy.ndarray":
            v = v.replace('[','').replace(']','')
            try:
                v = np.fromstring(v, dtype=float, sep=" ")
            except Exception as e:
                print("Error converting numpy.ndarray string back to array")
                print(e)
        else:
            raise TypeError("Unexpected data type (" + t + ") for field: " + f)
                
        setattr(mp_stack.modules[modIdx], f, v)
        
    mp_stack.evaluate(start=modIdx)
    
    # TODO: Only need to update plots starting at modIdx, figure out a good way
    #       to specify this offset in js.
    stackmods = mp_stack.modules[1:]
    plots = []
    for m in stackmods:
        p = plt.figure(figsize=FIGSIZE)
        m.do_plot(m)
        html = mpld3.fig_to_html(p)
        plots.append(html)
        plt.close(p)
    
    return jsonify(plots=plots)
    

def convert(string, type_):
    """Adapted from stackoverflow. Not quite working as desired.
    Stopped using for now but wanted to retain for later use.
    
    """
    import importlib
    import builtins
    try:
        cls = getattr(builtins, type_)
    except AttributeError:
        module, type_ = type_.rsplit(".", 1)
        module = importlib.import_module(module)
        cls = getattr(module, type_)
    return cls(string)
    

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
    
    #TODO: Not currently working, html components disabled for now
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
    # TODO: all_idx not getting picked up by form submission? disabled for now
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


@app.route('/get_kw_defaults')
def get_kw_defaults():
    """DEPRECATED"""
    
    global mp_stack
    kw = request.args.get('kSelected')
    mod = request.args.get('modAffected')
    module = getattr(nm, mod)
    editable_fields = module.user_editable_fields
    
    # TODO: still getting ndarray type error when trying to jsonify.
    #       looks like it's b/c there are some times multi-dim nd arrays
    #       as editable fields, which can't be serialized.
    #if hasattr(nm, kw):
    #    # if above passed w/o exception, kw must have been the
    #    # base class default
    #    inst = module(mp_stack)
    #    defaults = {
    #            field: getattr(inst, field)
    #            for field in editable_fields
    #            }
    #    # convert any np types to scalar equivalents
    #    for key in defaults.keys():
    #        x = defaults[key]
    #        if isinstance(x, float) or isinstance(x, int) or isinstance(x, str):
    #            pass
    #        elif isinstance(x, list):
    #            #flatten list into string
    #            defaults[key] = ','.join(x)
    #        else:
    #            try:
    #                x = defaults[key][0]
    #            except:
    #                x = defaults[key]
    #            finally:
    #                try:
    #                    x = np.asscalar(x)
    #                except Exception as e:
    #                    print("couldn't convert to np.scalar")
    #                    print(type(x))
    #                   print(e)
    #else:
    if hasattr(nk, kw):
        fn = getattr(nk, kw)
        if isinstance(fn, list):
            defaults = {
                field: value for value, field
                in enumerate(editable_fields)
                }
        else:
            defaults = get_kw_args(fn, editable_fields)
    else:
        print("Default values could not be found for: " + kw)
        defaults = {
                field: value for value, field
                in enumerate(editable_fields)
                }
        
    return jsonify(**defaults)
   

def get_kw_args(function, editables):
    """DEPRECATED"""
    source = inspect.getsource(function)
    values = {}
    for field in source:
        # find index of first character of field arg
        i = source.find(field + "=")
        # find index of first character of assigned value
        # IMPORTANT: this assumes that field and value are separated by
        #            single '=' character.
        j = i + len(field) + 2
        # get the last index of the assigned value
        k = source[j:].find(',')
        if k == -1:
            k = source[j:].find(')')
        # get the value
        values[field] = source[j:k+1]
        
    return values
    
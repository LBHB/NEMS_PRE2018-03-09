"""Placeholder views function file for modelpane.

Only used for testing the template right now.

"""
import itertools

from flask import request, render_template, Session, Response
import matplotlib.pyplot as plt, mpld3

from lib.baphy_utils import get_kw_file, get_celldb_file
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
    
    #session = Session()
    
    # TODO: Get these from results table, or from selectors?
    bSelected = request.form.get('batch')[:3]
    cSelected = request.form.getlist('celllist')[0]
    mSelected = request.form.getlist('modelnames')[0]
    
    # Get data filepaths from database (sCellFile?) and open them
    #paths = (session.query(SomeTable.someColumns)
    #        .filter(Table.cellid == cSelected)
    #        .filter(Table.modelname == mSelected).all()
    
    # Read data to find out which modules should be loaded, then load them
    # (import?)
    #for module in data.modules
    #    load/import module?
    #    how is this going to stay open after the request is returned?
    #    Maybe need some global variables available to all of the modelpane
    #    view functions, so that AJAX can interact with them?
    #    Flask.g the right place to store this?
    
    
    # Test code based on nems/misc/Test_Files/nems_test.py
    #stack = nm.nems_stack()
    #stack.meta['batch']=291     #bSelected
    #stack.meta['cellid']='bbl031f-a1'        #cSelected
    
    #keywords = "fb18ch100_fir10_fit00".split("_") 
    #keywords = mSelected.split("_")
    
    #TODO: don't need to do this anymore? looks like nems_mod takes care of it
    #if keywords[0] in "list of special loader modules":
        # pop out first keyword
    #loader = keywords.pop(0) # will move inside if statement
    
    #if loader is not None:
    #    load = getattr(nk,loader)
    #    load(stack)
    #else:
        # TODO: what should be used as default?
    #    nk.fb18ch100(stack)
        
    # TODO: What determines which module is added here?
    
    # TODO: this code for fitting, just want to load for modelpane
    #stack.append(nm.standard_est_val)    
        
    #for k in keywords:
    #    m = getattr(nk,k)
    #    m(stack)
        # After each module is added, need to get plot data some how?
        # Or does the stack need to be fit first?
        # How to pass plot data as JSON or simliar instead of posting
        # to ipython console?
        
    #stack.fitter = nf.basic_min(stack)
    #stack.fitter.maxit = 10
    #stack.fitter.do_fit()
    
    #stack.quick_plot()
    #embed = mpld3.fig_to_html(stack.quick_plot_html())
    #print(embed)
    
    # test code
    #i = 5
    #logo = (
    #        '/auto/users/jacob/nems/web/'
    #        'nems_analysis/static/lbhb_logo.jpg'
    #        )
    #logos = list(itertools.repeat(logo,i))
    #testMods = ['mod%s'%j for j in range(i)]
    #testPlots = [open_plot(j) for j in logos]
    #testPlots = ['plot%s'%j for j in range(i)]

    #bSelected = 291     #bSelected
    #cSelected = 'bbl031f-a1'        #cSelected
    #mSelected =  "fb18ch100_fir10_fit00"

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

    #session.close()

    return render_template(
            "/modelpane/modelpane.html", 
            modules=[m.name for m in stack.modules[1:]],
            plots=plots,
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
    
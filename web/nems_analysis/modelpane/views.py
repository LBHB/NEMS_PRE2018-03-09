"""Placeholder views function file for modelpane.

Only used for testing the template right now.

"""

from flask import render_template, Session

from nems_analysis import app

@app.route('/modelpane')
def modelpane_view():
    """Launches Modelpane window to inspect the model fit to the selected
    cell and model."""
    
    #session = Session()
    
    # TODO: Get these from results table, or from selectors?
    #cSelected = request.args.get('cSelected')
    #mSelected = request.args.get('mSelected')
    
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
    
    testMods = ['mod%s'%i for i in range(5)]
    testPlots = ['plot%s'%i for i in range(5)]
    
    #session.close()
    
    return render_template(
            "/modelpane/modelpane.html", modules=testMods,
            plots=testPlots
            )
    

@app.route('/update_modelpane')
def update_modelpane():
    """Placeholder functions. Update some parameter/attribute of a module?"""
    
    # Triggered if a selection is made in modelpane. Get the new specification
    # and add it to/overwrite it in the appropriate module.
    #modAffected = request.args.get('modAffected')
    #modSpec = request.args.get('specChanged')
    #modValue = request.args.get('newSpecValue')
    
    #modObject = Flask.g['modulesDict'][modAffected]
    #modObject.setattr(modSpec, modValue)
    #plot = modObject.make_new_plot_with_changed_value
    
    #return jsonify(plot=plot)
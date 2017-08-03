from flask import redirect, Response, url_for

from nems.web.nems_analysis import app
import nems.keywords as nk

@app.route('/cell_details/<cellid>')
def cell_details(cellid):
    return Response('Details for cellid: ' + cellid)
    # test code below this, won't run until response removed
    # TODO: parse cellid into animal info via gCellMaster?
    #       info should all be there already, just need to figure out the
    #       correct combination of queries.
    
    
@app.route('/model_details/<modelname>')
def model_details(modelname):
    #return Response('Details for modelname: ' + modelname)
    # test code below this, won't run until response removed
    keyword_list = modelname.split('_')
    kw_funcs = [
            getattr(nk, kw)
            if hasattr(nk, kw)
            else "Couldn't find keyword: {0}".format(kw)
            for kw in keyword_list
            ]
    docs = [
            func.__doc__ + '              '
            if func.__doc__
            else 'blank doc string'
            for func in kw_funcs
            ]
        
    return Response('         '.join(docs))
    # TODO: parse the stack.append() lines inside each kw function
    #       into a description of which module(s) is/are added and which
    #       arguments are specified
    #       ex: fb18ch100 should be translated to:
    #           load file with baphy_utils, args: fs=100,stimfmt='ozgf',chancount=18
    #           add load_mat with args: fs=100
    #           add crossval with args: defaults
    #       alternatively, just specify all of this in docstring for the func
    #       then pull out the doc strings here? probably much easier.
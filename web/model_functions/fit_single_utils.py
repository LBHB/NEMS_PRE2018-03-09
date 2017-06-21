""" helper functions and classes for internal use by fit_single_model 
or q_fit_single_model

These are packaged separately so that they can be called by q_fit_single_model 
via model queue or command line without depending on a running app server.

"""
    
import datetime

import numpy as np

def cleanup_file_string(s):
    """Remove curly braces and commas from string, then return string.
    ?DEPRECATED?
    """
    s = s.replace('{','')
    s = s.replace('}','')
    s = s.replace(',','')
    
    return s

class MultiResultError(Exception):
    """Throw this if more than one result is found when checking for an
    existing entry in NarfResults
    ?DEPRECATED?
    """
    pass
        
def fetch_meta_data(stack, r, attrs):
    """Assign attributes from model fitter object to NarfResults object.
    
    Arguments:
    ----------
    stack : nems_modules.stack
        Stack containing meta data, modules, module names et cetera
        (see nems_modules).
    r : sqlalchemy ORM object instance
        NarfResults object, either a blank one that was created before calling
        this function or one that was retrieved via a query to NarfResults.
        
    Returns:
    --------
    Nothing. Attributes of 'r' are modified in-place.
        
    """
    
    r.lastmod = datetime.datetime.now().replace(microsecond=0)
    
    for a in attrs:
        # list of non-numerical attributes, should be blank instead of 0.0
        if a in ['modelpath', 'modelfile', 'githash']:
            default = ''
        else:
            default = 0.0
        # TODO: hard coded fix for now to match up stack.meta names with 
        # narfresults names.
        # Either need to maintain hardcoded list of fields instead of pulling
        # from NarfResults, or keep meta names in fitter matched to columns
        # some other way if naming rules change.
        if 'fit' in a:
            k = a.replace('fit','est')
        elif 'test' in a:
            k = a.replace('test','val')
        else:
            k = a
        setattr(r, a, _fetch_attr_value(stack, k, default))


def _fetch_attr_value(stack,k,default=0.0):
    """Return the value of key 'k' of stack.meta, or default. Internal use."""
    
    # if stack.meta[k] is a string, return it.
    # if it's an ndarray or anything else with indicies, get the first index;
    # otherwise, just get the value. Then convert to scalar if np data type.
    # or if key doesn't exist at all, return the default value.
    if k in stack.meta:
        if stack.meta[k]:
            if not isinstance(stack.meta[k], str):
                try:
                    v = stack.meta[k][0]
                except:
                    v = stack.meta[k]
                finally:
                    try:
                        v = np.asscalar(v)
                    except:
                        pass
            else:
                v = stack.meta[k]
    else:
        v = default
        
    
    return v
    

def db_get_scellfiles(session,cellid,batch):
    """?DEPRECATED?"""
    # DEPRECATED -- now just passing est/val idents instead
    # if decide to use again in the future, would need to import narf batches
    # and scellfile,
    # which would break independence from app for q_fit_single
    #######################################################
    
    idents = session.query(NarfBatches.est_set,NarfBatches.val_set).filter\
                            (NarfBatches.cellid == cellid).filter\
                            (NarfBatches.batch == batch).all()
    # result should be a list of 2 items - one est_set and one val_set
    if len(idents) > 2:
        return ('error: more than one','set of idents for cell + batch')
    
    if type(idents[0]) is list:
        est_idents = [ident.replace('_est','') for ident in idents[0]]
    else:
        est_idents = [idents[0].replace('_est','')]
    if type(idents[1]) is list:
        val_idents = [ident.replace('_val','') for ident in idents[1]]
    else:
        val_idents = [idents[1].replace('_val','')]

    est_paths = []
    for est in est_idents:
        est_paths += session.query(sCellFile.stimfile,sCellFile.respfile).filter\
                                (sCellFile.cellid.ilike(cellid)).filter\
                                (sCellFile.stimfile.ilike(est)).filter\
                                (sCellFile.respfile.ilike(est)).all()

    val_paths = []
    for val in val_idents:
        val_paths += session.query(sCellFile.stimefile,sCellFile.respfile).filter\
                                (sCellFile.cellid.ilike(cellid)).filter\
                                (sCellFile.stimfile.ilike(val)).filter\
                                (sCellFile.respfile.ilike(val)).all()

    return (est_paths,val_paths)
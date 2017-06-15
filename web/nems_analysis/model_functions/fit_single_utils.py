""" helper functions and classes for internal use by fit_single_model 
or q_fit_single_model

These are packaged separately so that they can be called by q_fit_single_model 
via model queue or command line without depending on a running app server.

"""
    
import datetime

def cleanup_file_string(s):
    """Remove curly braces and commas from string, then return string."""
    s = s.replace('{','')
    s = s.replace('}','')
    s = s.replace(',','')
    
    return s

class MultiResultError(Exception):
    """Throw this if more than one result is found when checking for an
    existing entry in NarfResults
    
    """
    pass
        
def fetch_meta_data(mf,c,b,m,r):
    """Assign attributes from model fitter object to NarfResults object.
    
    Arguments:
    ----------
    mf : FERReT object instance
        FERReT (model-fitter) object instance that has loaded and fit its
        module queue, then assigned the resulting performance scores as
        attributes.
    c : string
        cellid that was passed to the FERReT object instance.
    b : string
        batch number that was passed to the FERReT object instance.
    m : string
        modelname that was passed to the FERReT object instance.
    r : sqlalchemy ORM object instance
        NarfResults object, either a blank one that was created before calling
        this function or one that was retrieved via a query to NarfResults.
        
    Returns:
    --------
    r : sqlalchemy ORM object
        A new NarfResults object instance with attribute values populated from 
        the corresponding attributes of the FERReT object instance.
        
    """
    
    #passed in
    r.cellid = c
    r.batch = b
    r.modelname = m
    #current datetime
    r.lastmod = datetime.datetime.now().replace(microsecond=0)
    
    #should be saved 
    r.r_fit = fetch_attr_value(mf,'r_fit')
    r.r_test = fetch_attr_value(mf,'r_test')
    r.r_test_rb = fetch_attr_value(mf,'r_test_rb')
    r.r_ceiling = fetch_attr_value(mf,'r_ceiling')
    r.r_floor = fetch_attr_value(mf,'r_floor')
    r.r_active = fetch_attr_value(mf,'r_active')
    r.mi_test = fetch_attr_value(mf,'mi_test')
    r.mi_fit = fetch_attr_value(mf,'mi_fit')
    r.nlogl_test = fetch_attr_value(mf,'nlogl_test')
    r.nlogl_fit = fetch_attr_value(mf,'nlogl_fit')
    r.mse_test = fetch_attr_value(mf,'mse_test')
    r.mse_fit = fetch_attr_value(mf,'mse_fit')
    r.cohere_test = fetch_attr_value(mf,'cohere_test')
    r.cohere_fit = fetch_attr_value(mf,'cohere_fit')
    r.n_parms = fetch_attr_value(mf,'n_parms',default=0)
    r.score = fetch_attr_value(mf,'score')
    r.sparsity = fetch_attr_value(mf,'sparsity')
    r.modelpath = fetch_attr_value(mf,'modelpath',default='')
    r.modelfile = fetch_attr_value(mf,'modelfile',default='')
    r.githash = fetch_attr_value(mf,'githash',default='')
    r.figurefile = fetch_attr_value(mf,'figurefile',default='')
    
    return r


def fetch_attr_value(mf,a,default=0.0):
    """Return the value of attribute 'a' of FERReT instance 'mf', or default"""
    if hasattr(mf,a):
        if getattr(mf,a) is not None:
            v = getattr(mf,a)
    else:
        v = default
        
    return v
    

def db_get_scellfiles(session,cellid,batch):
    
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
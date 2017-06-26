"""Functions for interacting with model fitter and model queuer.

These functions are designed to be called by model_functions.views after
a user clicks on Fit Single Model or Enqueue Models in the nems_analysis
web interface - for command-line usage, see q_fit_single_model.py.
Supporting functions are contained in fit_single_utils.py

Functions:
----------
fit_single_model:
    Instantiate model-fitting object a single time and
    write results to NarfResults, then return a summary to view function.

enqueue_models:
    Form a string of commands necessary to run q_fit_single_model
    from the command prompt for each cell+model combination for a
    given analysis (selected by the user in the web interface).
    Then write each entry to tQueue to be run via the model queuer.

"""
import datetime
import sys

from sqlalchemy import inspect
import pandas as pd
import pandas.io.sql as psql
import numpy as np

from nems_analysis import Session, tQueue, NarfResults, NarfBatches
from .fit_single_utils import (
        cleanup_file_string, fetch_meta_data, MultiResultError,
        )
from nemsclass import FERReT
from lib.baphy_utils import get_celldb_file, get_kw_file

#TODO: will need to put a similar version of fit_single in a separate .py
#       for use with enqueue models. Needs to not import anything from the app
#       so that it can be run from command line without launching the rest
#       of the package. Can just copy paste code except will need to make its own
#       db connection instead of importing Session from nems_analysis

    
def enqueue_models(celllist,batch,modellist):
    """Call enqueue_single_model for every combination of cellid and modelname
    contained in the user's selections.
    
    Arguments:
    ----------
    celllist : list
        List of cellid selections made by user.
    batch : string
        batch number selected by user.
    modellist : list
        List of modelname selections made by user.
    
    Returns:
    --------
    data : string  <-- not yet finalized
        Some message indicating success or failure to the user, to be passed
        on to the web interface by the calling view function.
        
    See Also:
    ---------
    . : enqueue_single_model
    Narf_Analysis : enqueue_models_callback
    
    """
    # Not yet ready for testing - still need to coordinate the supporting
    # functions with the model queuer.
    #for model in modellist:
        #for cell in celllist:
            #enqueue_single_model(cell,batch,model)
    
    data = 'Placeholder success/failure messsage for user.'
    return data


def enqueue_single_model(cellid,batch,modelname):
    """Not yet developed, likely to change a lot.
    
    Returns:
    --------
    tQueueId : int
        id (primary key) that was assigned to the new tQueue entry, or -1.
        
    See Also:
    ---------
    Narf_Analysis : enqueue_single_model
    
    """

    session = Session()
    tQueueId = -1
    
    # TODO: what needs to go here so that queuer knows to run this
    #       as a python script?
    commandPrompt = "fit_single_model(%s,%s,%s,autoplot=False)"%(
                                     cellid,batch,modelname,
                                     )

    note = "%s/%s/%s"%(cellid,batch,modelname)
    
    #query tQueue to check if entry with same cell/batch/model already exists
    result = session.query(tQueue).filter(tQueue.note == note).all()
    
    # if it does, check its 'complete' status and take different action based on
    # status
    
    #if (result) and result['complete'] <= 0:
        #TODO:
        #incomplete entry for note already exists, skipping
        #update entry with same note? what does this accomplish?
        #moves it back into queue maybe?
        
    #elif (result) and result['complete'] == 2:
        #TODO:
        #dead queue entry for note exists, resetting
        #update complete and progress status each to 0
        #what does this do? doesn't look like the sql is sent right away,
        #instead gets assigned to [res,r]
        
    #elif (result) and result['complete'] == 1:
        #TODO:
        #resetting existing queue entry for note
        #update complete and progress status each to 0
        #same as above, what does this do?
        
    #else (result must not have existed? or status value was greater than 2)
        # add new entry
        # job = tQueue()
        # session.add(add_model_to_queue(commandPrompt,note,job))
    
    #session.commit()
    #tQueueId = job.id
    session.close()
    #return tQueueId
    
def add_model_to_queue(commandPrompt,note,job,priority=1,rundataid=0):
    """Not yet developed, likely to change a lot.
    
    Returns:
    --------
    job : tQueue object instance
        tQueue object with variables assigned inside function based on
        arguments.
        
    See Also:
    ---------
    Narf_Analysis: dbaddqueuemaster
    
    """
    
    #TODO: does user need to be able to specificy priority and rundataid somewhere
    #       in web UI?
    
    #TODO: why is narf version checking for list vs string on prompt and note?
    #       won't they always be a string passed from enqueue function?
    #       or want to be able to add multiple jobs manually from command line?
    #       will need to rewrite with for loop to to add this functionality in the
    #       future if desired
    
    #TODO: set these some where else? able to choose from UI?
    #       could grab user name from login once implemented
    user = 'default-user-name-here?'
    progname = 'python queuerun'
    allowqueuemaster=1
    waitid = 0
    dt = str(datetime.datetime.now().replace(microsecond=0))
    
    job.rundataid = rundataid
    job.progname = progname
    job.priority = priority
    job.parmstring = commandPrompt
    job.queuedate = dt
    job.allowqueuemaster = allowqueuemaster
    job.user = user
    job.note = note
    job.waitid = waitid
    
    return job





class dat_object():
    """ placeholder for data array until model fitter linked up,
    for testing only.
    
    """
    def __init__(self):
        pass


def fit_single_model(cellid,batch,modelname):
    """Instantiate model-fitting object a single time and write the results to 
    NarfResults, then return a summary to view function.
    
    > DEPRECATED -- see lib.nems_main.fit_single_model <
    
    Arguments:
    ----------
    cellid : string
        cellid that was passed to the view function by user selection.
    batch : string
        batch number that was passed to the view function by user selection.
    modelname : string
        modelname that was passed to the view function by user selection.
        
    Returns:
    --------
    data : dict-like
        Data structure containing summary information for the results of the
        model fitting procedure. Final format not set, but likely to include
        some hand-picked performance measures like r_test and/or a preview
        image containing STRF, spike raster, etc.
    
    See Also:
    ---------
    fit_single_utils : cleanup_file_string, fetch_meta_data, MultiResultError
    
    """
    
    session = Session()
    # TODO: don't worry about these for now, may change
    # assigned like a list comprehension for syntax
    # but should only be one result from DB
    #filecodes = [
    #            code[0] for code in \
    #            session.query(NarfBatches.filecodes)\
    #            .filter(NarfBatches.cellid == cellid)\
    #            .filter(NarfBatches.batch == batch).all()
    #            ]
    
    
    # DEPRECATED
    # get filepaths for both est and val data sets
    # est_set_files,val_set_files = db_get_scellfiles(session,cellid,batch)
    
    
    # TODO: Are the identifiers still needed? doesn't look like the file
    #       loader util is actually doing anything with them.
    idents = (
            session.query(NarfBatches.est_set,NarfBatches.val_set)
            .filter(NarfBatches.cellid == cellid)
            .filter(NarfBatches.batch == batch).all()
            )
    # empty list should return false
    if not idents:
        # no file identifiers exist for cell/batch combo in NarfBatches
        # TODO: should this throw an error? or do something else?
        pass
    # First index gets sqlalchemy object returned by query,
    # second index gets the items from inside the object (one for each column)
    est_ident = idents[0][0]
    val_ident = idents[0][1]
    #TODO: should do string cleanup here or pass as-is to model fitter?
    est_ident = cleanup_file_string(est_ident)
    val_ident = cleanup_file_string(val_ident)


    # TODO: still calling FERReT or using nems_test variant instead?
    # pass cellid, batch, modelname, est_file_ident and val_file_ident to
    # model fitter object -- or just filepath instead?
    keywords = modelname.split('_')
    file = get_kw_file(batch,cellid,keywords.pop(0))
    keywords = '_'.join(keywords)
    ModelFitter = FERReT(filepath=file, imp_type='standard', keyword=keywords)
    #TODO: got to 'creating validation and training sets' inside FERReT object
    #           before error. Stopping testing until decided if we're keeping
    #           this procedure.
    
    
    
    # tell model fitter to run the queue of modules
    # returns nothing
    ModelFitter.run_fit()
    # get 3d numpy array from Model Fitter after modules run
    # each returns a dict of numpy arrays with 
    # keys 'stim, resp, pup, and predicted'
    est_data = ModelFitter.apply_to_est()
    val_data = ModelFitter.apply_to_val()
    
    print("est_data retrieved?"); print(est_data)
    print("val_data retrieved?"); print(val_data)
    # TODO: save array(s) to file(s) appropriately and return filepath(s)
    # Calculate a filepath based on some rules? or just re-use get_kw_file?
    #np.save(est_filepath, est_data)
    #np.save(val_filepath, val_data)
    # Write the filepaths to the database. Where should they go?
    #tableEntry = (
    #        session.query(SomeTableName.columnsForFilePath)
    #        .filter(cellid == cellid)
    #        .filter(modelname == modelname)
    #        .all()
    #        )
    #tableEntry.estCol = est_filepath
    #tableEntry.valCol = val_filepath
    
    # dat_object() for testing only
    data_array = dat_object()
    data_array.r_test = 0.5
    data_array.n_parms = 1
    data_array.data = "placeholder for model fitter until model fitter linked up"
    
    check_exists = (
            session.query(NarfResults)
            .filter(NarfResults.cellid == cellid)
            .filter(NarfResults.batch == batch)
            .filter(NarfResults.modelname == modelname)
            .all()
            )
    
    if len(check_exists) == 0:
        # If no entry exists in narf results for cell/model/batch combo,
        # write in a new entry.
        r = NarfResults()
        r = fetch_meta_data(data_array,cellid,batch,modelname,r)
        session.add(r)
    elif len(check_exists) == 1:
        # If one entry exists in NarfResults, overwrite it.
        r = check_exists[0]
        r = fetch_meta_data(data_array,cellid,batch,modelname,r)
    else:
        # If more than one entry exists in NarfResults, something went wrong
        raise MultiResultError(
                "Multiple entries in Narf Results for cell: %s,"
                "batch: %s, modelname: %s"%(cellid,batch,modelname)
                )

    # Test to make sure attributes are being correctly assigned from
    # metadata in array
    print("Printing attributes assigned to results entry:")
    mapper = inspect(r)
    for c in mapper.attrs:
        print(c.key)
        print(getattr(r,c.key))
        
    # Leave session.commit() commented out until ready to test with database
    # IF LEFT IN, THIS WILL SAVE TO / OVERWRITE DATA IN CellDB
    #session.commit()
    session.close()
    
    data = data_array
    
    # TODO: Only need to return data that will be displayed to user, 
    # i.e. figurefile or path and a results summary. Pass as dict?
    # i.e. {'figurefile':'/auto/user/data...','r_fit':0.48,....}
    # or something to that effect
    return data


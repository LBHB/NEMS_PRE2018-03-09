"""
functions/classes for handling interactions with model fitting

fit_single_model instantiates model-fitting object a single time and
writes results to NarfResults.

enqueue_models forms a string of commands necessary to run fit_single_model
from the command prompt for each cell+model combination for a given analysis,
then writes each entry to tQueue to be run via model queue.

TODO:
class/object approach necessary here? really just carrying out the same
procedure each time. Going with simple functions for now.
"""
import datetime
import sys

from sqlalchemy import inspect
import pandas as pd
import pandas.io.sql as psql

from nems_analysis import Session, tQueue, NarfResults, NarfBatches
from .fit_single_utils import cleanup_file_string, fetch_meta_data, MultiResultError
sys.path.insert(0,'/auto/users/jacob/nems')
from modNEM import FERReT


#TODO: restructure nems package to avoid this issue


#TODO: will need to put a similar version of fit_single in a separate .py
#       for use with enqueue models. Needs to not import anything from the app
#       so that it can be run from command line without launching the rest
#       of the package. Can just copy paste code except will need to make its own
#       db connection instead of importing Session from nems_analysis


class dat_object():
    """ placeholder for data array until model fitter linked up,
    for testing only 
    
    """
    def __init__(self):
        pass


def fit_single_model(cellid,batch,modelname):
    
    session = Session()
    
    """
    # TODO: don't worry about these for now, may change
    # assigned like a list comprehension for syntax
    # but should only be one result from DB
    filecodes = [
                code[0] for code in \
                session.query(NarfBatches.filecodes)\
                .filter(NarfBatches.cellid == cellid)\
                .filter(NarfBatches.batch == batch).all()
                ]
    """
    
    """
    # DEPRECATED
    # get filepaths for both est and val data sets
    est_set_files,val_set_files = db_get_scellfiles(session,cellid,batch)
    """
    
    idents = session.query(NarfBatches.est_set,NarfBatches.val_set)\
                          .filter(NarfBatches.cellid == cellid)\
                          .filter(NarfBatches.batch == batch).all()
    
    if len(idents) == 0:
        # no file identifiers exist for cell/batch combo in NarfBatches
        # TODO: should this throw an error? or do something else?
        pass
    
    est_ident = idents[0][0]
    val_ident = idents[0][1]
    
    #TODO: should do string cleanup here or pass as-is to model fitter?
    est_ident = cleanup_file_string(est_ident)
    val_ident = cleanup_file_string(val_ident)
    
    print("estimation file identifer(s) and validation file identifier(s):")
    print(est_ident)
    print(val_ident)
    
    # pass cellid, batch, modelname, est_file_ident and val_file_ident to
    # model fitter object
    # TODO: supposed to have the option of directly passing a file instead?
    
    #ModelFitter = FERReT(cellid, batch, modelname, est_ident, val_ident)
    
    # tell model fitter to run the queue of modules
    #ModelFitter.run_fit()
    
    # get 3d numpy array from Model Fitter after modules run
    # TODO: looks like this has changed? what do we need to call now?
    #       ~apply to fit
    #       and ~apply to est?
    #ModelFitter.assemble_data_array()

    # TODO: save array(s) to file(s) appropriately and return filepath(s)
    
    # TODO: need some kind of timeout warning for user?
    
    # dat_object() for testing only
    data_array = dat_object()
    data_array.r_test = 0.5
    data_array.n_parms = 1
    data_array.data = "placeholder for model fitter until model fitter linked up"
    
    check_exists = session.query(NarfResults).filter\
                    (NarfResults.cellid == cellid).filter\
                    (NarfResults.batch == batch).filter\
                    (NarfResults.modelname == modelname).all()
    
    if len(check_exists) == 0:
        # if no entry in narf results for cell/model/batch combo, write in new entry
        r = NarfResults()
        r = fetch_meta_data(data_array,cellid,batch,modelname,r)
        session.add(r)
        
    elif len(check_exists) == 1:
        # if one entry in narf results, overwrite it
        r = check_exists[0]
        r = fetch_meta_data(data_array,cellid,batch,modelname,r)
        
    else:
        # if more than one entry exists, something went wrong
        raise MultiResultError("Multiple entries in Narf Results for cell: %s,\
                               batch: %s, modelname: %s"%(cellid,batch,modelname))


    # leave session.commit() commented out until ready to test with database
    # IF LEFT IN, THIS WILL SAVE TO / OVERWRITE DATA IN DB
    
    #session.commit()
    
    # test to make sure attributes are being correctly assigned from
    # metadata in array
    print("Printing attributes assigned to results entry:")
    mapper = inspect(r)
    for c in mapper.attrs:
        print(c.key)
        print(getattr(r,c.key))
    
    
    session.close()
    
    data = data_array
    
    # only need to return data that will be displayed to user, i.e. figurefile
    # or path and a results summary. Pass as dict?
    # i.e. {'figurefile':'/auto/user/data...','r_fit':0.48,....}
    # or something to that effect
    return data

    
def enqueue_models(celllist,batch,modellist):
    # See narf_analysis --> enqueue models callback
    #for model in modellist:
        #for cell in celllist:
            #enqueue_single_model(cell,batch,model)
    
    data = 'some kind of success/failure messsage for user'
    return data


def enqueue_single_model(cellid,batch,modelname):
    # See narf_analysis --> enqueue_single_model
    session = Session()
    commandPrompt = "q_fit_single_model(%s,%s,%s)"%(\
                                     cellid,batch,modelname)

    note = "%s/%s/%s"%(cellid,batch,modelname)
    
    #query tQueue to check if entry with same cell/batch/model already exists
    result = psql.read_sql_query(session.query(tQueue).filter\
                                 (tQueue.note == note).statement,session.bind)
    
    # if it does, check its 'complete' status and take different action based on
    # status
    
    #if result is not empty and result['complete'] <= 0:
        #incomplete entry for note already exists, skipping
        #update entry with same note? what does this accomplish?
        #moves it back into queue maybe?
        
    #elif result is not empty and result['complete'] == 2:
        #dead queue entry for note exists, resetting
        #update complete and progress status each to 0
        #what does this do? doesn't look like the sql is sent right away,
        #instead gets assigned to [res,r]
        
    #elif result is not empty and result['complete'] == 1:
        #resetting existing queue entry for note
        #update complete and progress status each to 0
        #same as above, what does this do?
        
    #else (result must not have existed? or status value was greater than 2)
        # add new entry
        # sql = add_model_to_queue(commandPrompt,note)
        # session.add(tQueue(rundataid=sql['rundataid'],progname=sql['progname'],\
        #                    priority=sql['priority'],parmstring=sql['parmstring'],\
        #                    queuedate=sql['queuedate'],allowqueuemaster=\
        #                    sql['allowqueuemaster'],user=sql['user'],note=\
        #                    sql['note'],waitid=sql['waitid']))
    
    session.close()
    
def add_model_to_queue(commandPrompt,note,priority=1,rundataid=0):
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
    
    sqlEntries = {'rundataid':rundataid,'progname':progname,\
                  'priority':priority,'parmstring':commandPrompt,'queuedate':dt,\
                  'allowqueuemaster':allowqueuemaster,'user':user,'note':note,\
                  'waitid':waitid}
    
    return sqlEntries
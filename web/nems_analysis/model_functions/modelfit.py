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

from nems_analysis import Session, tQueue, NarfResults, NarfBatches, sCellFile
import pandas as pd
import pandas.io.sql as psql
import datetime

from modNEM import FERReT

#TODO: will need to put a similar version of fit_single in a separate .py
#       for use with enqueue models. Needs to not import anything from the app
#       so that it can be run from command line without launching the rest
#       of the package. Can just copy paste code except will need to make its own
#       db connection instead of importing Session from nems_analysis

def fit_single_model(cellid,batch,modelname):
    session = Session()
        
    keywords = modelname.split('_')
    
    # assigned like a list comprehension for syntax but should only be one result from DB
    filecodes = [code[0] for code in session.query(NarfBatches.filecodes).filter\
                (NarfBatches.cellid == cellid).filter\
                (NarfBatches.batch == batch).all()]
    
    # get filepaths for both est and val data sets
    # TODO: currently these return the filenames, but not the path leading to them
    est_set_files,val_set_files = db_get_scellfiles(session,cellid,batch)
    
    # pass file paths, behavior codes
    ModelFitter = FERReT(est_files=est_set_files,val_files=val_set_files,\
                         behavior=filecodes,queue=keywords)
    
    # tell model fitter to run the queue of modules
    ModelFitter.run_fit()
    
    # get 3d numpy array from Model Fitter after modules run
    ModelFitter.assemble_data_array()

    # TODO: save array(s) to file(s) appropriately and return filepath(s)
    
    # TODO: need some kind of timeout warning for user?
    
    # form a pd.Series for NarfResults --> cellid=cellid, batch=batch,modelname=modelname,
    #                       other fields pulled from ModelObject.fieldAttribute, filepaths
    #                       fields retrieved above.
    
    
    """
    query NarfResults with cellid, batch and modelname - if get a result, then
    an entry for this cell + batch + model combination already exists, so need
    to delete it before adding new entry
    """
    
    """
    then add via sql alchemy explicitly
    
    session.add_all([
            Table(column1=data_from_modules['column1'], column2=data_from_modules['column2']),
            --repeat for each entry
            //construct list outside of this first for multiple entries
            ])
    
    session.commit()
    """
    
    """
    OR format entry within dataframe (since we're ultimately just adding 1 row)
    
    dataframe['name of series corresponding to desired values'].to_sql\
        (NarfResults,session.bind)
    """
    
    session.close()
    
    data = 'really cool model fitting data stuff'
    
    # only need to return data that will be displayed to user, i.e. figurefile
    # or path and a results summary. Pass as dict?
    # i.e. {'figurefile':'/auto/user/data...','r_fit':0.48,....}
    # or something to that effect
    return data

def db_get_scellfiles(session,cellid,batch):
    
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
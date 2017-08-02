#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nems_db library

Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import datetime

import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

try:
    import nems_config.AWS_Config as awsc
    AWS = awsc.Use_AWS
    if AWS:
        from nems.EC2_Mgmt import check_instance_count
except:
    AWS = False

# Database settings
# To specify database settings, store a file named Database_Info.py in the
# nems_config directory (inside the top-level package folder) with this format:
#   host = 'hosturl'
#   user = 'username'
#   passwd = 'password'
#   database = 'database'
# Order doesn't matter.
try:
    import nems_config.Database_Info as db
    db_uri = 'mysql+pymysql://{0}:{1}@{2}/{3}'.format(
                    db.user,db.passwd,db.host,db.database
                    )
except Exception as e:
    print('No database info detected')
    print(e)
    #db_uri = 'sqlite:////path/to/default/database/file'
    raise e

try:
    import nems_config.Cluster_Database_Info as clst_db
    # format:      dialect+driver://username:password@host:port/database
    # to-do default port = 3306
    if not hasattr(clst_db, 'port'):
        port = 3306
    else:
        port = clst_db.port
        
    clst_db_uri = 'mysql+pymysql://{0}:{1}@{2}:{3}/{4}'.format(
                        clst_db.user, clst_db.passwd, clst_db.host,
                        port, clst_db.database,
                        )
except Exception as e:
    print('No cluster database info detected')
    print(e)
    #clst_db_uri = 'sqlite:////path/to/default/database/file'
    raise e
    
# sets how often sql alchemy attempts to re-establish connection engine
# TODO: query db for time-out variable and set this based on some fraction of that
POOL_RECYCLE = 7200;

# create a database connection engine
engine = create_engine(db_uri, pool_recycle=POOL_RECYCLE)

#create base class to mirror existing database schema
Base = automap_base()
Base.prepare(engine, reflect=True)

NarfUsers = Base.classes.NarfUsers
NarfAnalysis = Base.classes.NarfAnalysis
NarfBatches = Base.classes.NarfBatches
NarfResults = Base.classes.NarfResults
tQueue = Base.classes.tQueue
tComputer = Base.classes.tComputer
sCellFile = Base.classes.sCellFile
sBatch = Base.classes.sBatch

# import this when another module needs to use the database connection.
# used like a class - ex: 'session = Session()'
Session = sessionmaker(bind=engine)


# Same as above, but duplicated for use w/ cluster
cluster_engine = create_engine(clst_db_uri, pool_recycle=POOL_RECYCLE)

cluster_Base = automap_base()
cluster_Base.prepare(cluster_engine, reflect=True)

cluster_tQueue = cluster_Base.classes.tQueue
cluster_tComputer = cluster_Base.classes.tComputer

cluster_Session = sessionmaker(bind=cluster_engine)


def enqueue_models(celllist, batch, modellist, force_rerun=False, user=None):
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
    pass_fail : list
        List of strings indicating success or failure for each job that
        was supposed to be queued.
        
    See Also:
    ---------
    . : enqueue_single_model
    Narf_Analysis : enqueue_models_callback
    
    """
    # Not yet ready for testing - still need to coordinate the supporting
    # functions with the model queuer.
    pass_fail = []
    for model in modellist:
        for cell in celllist:
            queueid, message = enqueue_single_model(
                        cell, batch, model, force_rerun, user
                        )
            if not queueid:
                pass_fail.append(
                        '\nFailure: {0}, {1}, {2}'
                        .format(cell, batch, model)
                        )
            else:
                pass_fail.append(
                        '\n queueid: {0},'
                        '\n message: {1}'
                        .format(queueid, message)
                        )
    
    # Can return pass_fail instead if prefer to do something with it in views
    print('\n'.join(pass_fail))
    return


def enqueue_single_model(cellid, batch, modelname, force_rerun, user):
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
    cluster_session = cluster_Session()
    
    # TODO: anything else needed here? this is syntax for nems_fit_single
    #       command prompt wrapper in main nems folder.
    commandPrompt = (
            "/auto/users/nems/anaconda3/bin/python "
            "/auto/users/nems/nems/nems_fit_single.py {0} {1} {2}"
            .format(cellid, batch, modelname)
            )

    note = "%s/%s/%s"%(cellid,batch,modelname)
    
    result = (
            session.query(NarfResults)
            .filter(NarfResults.cellid == cellid)
            .filter(NarfResults.batch == batch)
            .filter(NarfResults.modelname == modelname)
            .first()
            )
    if result and not force_rerun:
        print("Entry in NarfResults already exists for: %s, skipping.\n"%note)
        session.close()
        cluster_session.close()
        return -1, 'skip'
    
    #query tQueue to check if entry with same cell/batch/model already exists
    qdata = (
            cluster_session.query(cluster_tQueue)
            .filter(cluster_tQueue.note == note)
            .first()
            )
    
    # if it does, check its 'complete' status and take different action based on
    # status
    
    job = None
    message = None
    
    if qdata and (int(qdata.complete) <= 0):
        #TODO:
        #incomplete entry for note already exists, skipping
        #update entry with same note? what does this accomplish?
        #moves it back into queue maybe?
        message = "Incomplete entry for: %s already exists, skipping.\n"%note
        job = qdata
    elif qdata and (int(qdata.complete) == 2):
        #TODO:
        #dead queue entry for note exists, resetting
        #update complete and progress status each to 0
        #what does this do? doesn't look like the sql is sent right away,
        #instead gets assigned to [res,r]
        message = "Dead queue entry for: %s already exists, resetting.\n"%note
        qdata.complete = 0
        qdata.progress = 0
        job = qdata
    elif qdata and (int(qdata.complete) == 1):
        #TODO:
        #resetting existing queue entry for note
        #update complete and progress status each to 0
        #same as above, what does this do?
        message = "Resetting existing queue entry for: %s\n"%note
        qdata.complete = 0
        qdata.progress = 0
        job = qdata
    else:
        #result must not have existed, or status value was greater than 2
        # add new entry
        message = "Adding job to queue for: %s\n"%note
        job = add_model_to_queue(commandPrompt, note, user)
        cluster_session.add(job)
    
    queueid = job.id
    
    # don't need to commit the regular session since results don't change
    cluster_session.commit()
    cluster_session.close()
    session.close()
    
    if AWS:
        check_instance_count()
    else:
        pass

    return queueid, message
    
def add_model_to_queue(commandPrompt, note, user, priority=1, rundataid=0):
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
    
    job = cluster_tQueue()
    
    if user:
        user = user
    else:
        user = 'None'
    linux_user = 'nems'
    allowqueuemaster=1
    waitid = 0
    dt = str(datetime.datetime.now().replace(microsecond=0))
    
    job.rundataid = rundataid
    job.progname = commandPrompt
    job.priority = priority
    job.parmstring = ''
    job.queuedate = dt
    job.allowqueuemaster = allowqueuemaster
    job.user = user
    job.linux_user = linux_user
    job.note = note
    job.waitid = waitid
    
    return job

def update_job_complete(queueid):
    # mark job complete
    # svd old-fashioned way of doing
    #sql="UPDATE tQueue SET complete=1 WHERE id={}".format(queueid)
    #result = conn.execute(sql)
    #conn.close()
   
    cluster_session = cluster_Session()
    # also filter based on note? - should only be one result to match either
    # filter, but double checks to make sure there's no conflict
    #note = "{0}/{1}/{2}".format(cellid, batch, modelname)
    #.filter(tQueue.note == note)
    qdata = (
            cluster_session.query(cluster_tQueue)
            .filter(cluster_tQueue.id == queueid)
            .first()
            )
    if not qdata:
        # Something went wrong - either no matching id, no matching note,
        # or mismatch between id and note
        print("Invalid query result when checking for queueid & note match")
        print("/n for queueid: %s"%queueid)
    else:
        qdata.complete = 1
        cluster_session.commit()
       
    cluster_session.close()
    
def update_job_start(queueid):
    conn = cluster_engine.connect()
    # tick off progress, job is live
    sql = (
            "UPDATE tQueue SET complete=-1,progress=progress+1 WHERE id={}"
            .format(queueid)
            )
    return conn.execute(sql)

def save_results(stack, preview_file, queueid=None):
    session = Session()
    cluster_session = cluster_Session()
    
    cellid = stack.meta['cellid']
    batch = stack.meta['batch']
    modelname = stack.meta['modelname']
    
    # Can't retrieve user info without queueid, so if none was passed
    # use the default blank user info
    if queueid:
        job = (
                cluster_session.query(cluster_tQueue)
                .filter(cluster_tQueue.id == queueid)
                .first()
                )
        user = job.user
        narf_user = (
                session.query(NarfUsers)
                .filter(NarfUsers.username == user)
                .first()
                )
        labgroup = narf_user.labgroup
    else:
        user = ''
        labgroup = 'SPECIAL_NONE_FLAG'


    r = (
            session.query(NarfResults)
            .filter(NarfResults.cellid == cellid)
            .filter(NarfResults.batch == batch)
            .filter(NarfResults.modelname == modelname)
            .first()
            )
    collist = ['%s'%(s) for s in NarfResults.__table__.columns]
    attrs = [s.replace('NarfResults.', '') for s in collist]
    attrs.remove('id')
    attrs.remove('figurefile')
    attrs.remove('lastmod')
    if not r:
        r = NarfResults()
        r.figurefile = preview_file
        r.username = user
        if not labgroup == 'SPECIAL_NONE_FLAG':
            try:
                if not labgroup in r.labgroup:
                    r.labgroup += ', %s'%labgroup
            except TypeError:
                # if r.labgroup is none, ca'nt check if user.labgroup is in it
                r.labgroup = labgroup
        fetch_meta_data(stack, r, attrs)
        # TODO: assign performance variables from stack.meta
        session.add(r)
    else:
        r.figurefile = preview_file
        # TODO: This overrides any existing username or labgroup assignment.
        #       Is this the desired behavior?
        r.username = user
        if not labgroup == 'SPECIAL_NONE_FLAG':
            try:
                if not labgroup in r.labgroup:
                    r.labgroup += ', %s'%labgroup
            except TypeError:
                # if r.labgroup is none, can't check if labgroup is in it
                r.labgroup = labgroup
        fetch_meta_data(stack, r, attrs)
        
    results_id = r.id
    session.commit()
    session.close()
    
    return results_id

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
        
    
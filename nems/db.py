#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nems_db library

Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

import datetime

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
    db_uri = 'mysql+pymysql://%s:%s@%s/%s'%(
                    db.user,db.passwd,db.host,db.database
                    )
except Exception as e:
    print('No database info detected')
    print(e)
    db_uri = 'sqlite:////path/to/default/database/file'

try:
    import nems_config.Cluster_Database_Info as clst_db
    # format:      dialect+driver://username:password@host:port/database
    clst_db_uri = 'mysql+pymysql://%s:%s@%s:%s/%s'%(
                        clst_db.user, clst_db.passwd, clst_db.host,
                        clst_db.port, clst_db.database,
                        )
except Exception as e:
    print('No cluster database info detected')
    print(e)
    clst_db_uri = 'sqlite:////path/to/default/database/file'
    
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


def enqueue_models(celllist, batch, modellist, force_rerun=False):
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
            queueid = enqueue_single_model(cell, batch, model, force_rerun)
            if int(queueid) < 0:
                pass_fail.append(
                        'Failure: {0}, {1}, {2}'
                        .format(cell, batch, model)
                        )
            else:
                pass_fail.append(
                        'Success: {0}, {1}, {2} \n'
                        'added to queue with queue id: {3}'
                        .format(cell, batch, model, queueid)
                        )
    
    return pass_fail


def enqueue_single_model(cellid, batch, modelname, force_rerun):
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
    tQueueId = -1
    
    # TODO: anything else needed here? this is syntax for nems_fit_single
    #       command prompt wrapper in main nems folder.
    commandPrompt = (
            "python nems/nems_fit_single.py {0} {1} {3}"
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
        return -1
    
    #query tQueue to check if entry with same cell/batch/model already exists
    qdata = cluster_session.query(tQueue).filter(tQueue.note == note).all()
    
    # if it does, check its 'complete' status and take different action based on
    # status
    
    if qdata and (int(qdata.complete) <= 0):
        #TODO:
        #incomplete entry for note already exists, skipping
        #update entry with same note? what does this accomplish?
        #moves it back into queue maybe?
        print("Incomplete entry for: %s already exists, skipping.\n"%note)
        return -1
    elif qdata and (int(qdata.complete) == 2):
        #TODO:
        #dead queue entry for note exists, resetting
        #update complete and progress status each to 0
        #what does this do? doesn't look like the sql is sent right away,
        #instead gets assigned to [res,r]
        print("Dead queue entry for: %s already exists, resetting.\n"%note)
        qdata.complete = 0
        qdata.progress = 0
        return -1
    elif qdata and (int(qdata.complete) == 1):
        #TODO:
        #resetting existing queue entry for note
        #update complete and progress status each to 0
        #same as above, what does this do?
        print("Resetting existing queue entry for: %s\n"%note)
        qdata.complete = 0
        qdata.progress = 0
        return -1
    else: #result must not have existed? or status value was greater than 2
        # add new entry
        print("Adding job to queue for: %s\n"%note)
        job = cluster_tQueue()
        session.add(add_model_to_queue(commandPrompt,note,job))
    
    # uncomment session.commit() when ready to test saving to database
    session.commit()
    tQueueId = job.id
    session.close()
    
    if AWS:
        check_instance_count()
    else:
        pass
    
    return tQueueId
    
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
    progname = 'python3'
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
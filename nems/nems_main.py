#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:16:37 2017

@author: svd
"""
import numpy as np
import nems.nems_modules as nm
import nems.nems_stack as ns
import nems.nems_fitters as nf
import nems.nems_utils as nu
import nems.nems_keywords as nk
import nems.baphy_utils as baphy_utils
import os
import datetime
import copy
import scipy.stats as spstats

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

"""
fit_single_model - create, fit and save a model specified by cellid, batch and modelname

example fit on nice IC cell:
    import lib.nems_main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    nems.fit_single_model(cellid,batch,modelname)

"""
def fit_single_model(cellid, batch, modelname, autoplot=True,**xvals): #Remove xvals later, need to rework web app
    """
    Fits a single NEMS model. With the exception of the autoplot feature,
    all the details of modelfitting are taken care of by the model keywords.
    
    fit_single_model functions by iterating through each of the keywords in the
    modelname, and perfroming the actions specified by each keyword, usually 
    appending a nems module. If nested crossvla is specified in one of the modules,
    it will simply re-evaluate the keywords for each new estimation set, and store
    the fitted parameters for each module. Once all the estimation sets have been fit,
    fit_single_model then evaulates the validation datasets and saves the stack.
    It autoplot is true, it generates plots specified by each nems module. 
    
    fit_single_model returns the evaluated stack, which contains both the estimation
    and validation datasets. In the caste of nested crossvalidation, the validation
    dataset contains all the data, while the estimation dataset is just the estimation 
    data that was fitted last (i.e. on the last nest)
    """
    stack=ns.nems_stack()
    
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname
    
    # extract keywords from modelname    
    keywords=modelname.split("_")
    stack.cv_counter=0
    stack.cond=False
    stack.fitted_modules=[]
    while stack.cond is False:
        print('iter loop='+str(stack.cv_counter))
        stack.clear()
        stack.valmode=False
        for k in keywords:
            f = getattr(nk, k)
            f(stack)
            
       #TODO: this stuff below could be wrapped into do_fit somehow
        phi=[] 
        for idx,m in enumerate(stack.modules):
            this_phi=m.parms2phi()
            if this_phi.size:
                if stack.cv_counter==0:
                    stack.fitted_modules.append(idx)
                phi.append(this_phi)
        phi=np.concatenate(phi)
        stack.parm_fits.append(phi)

        if stack.nests==1:
            stack.cond=True
        else:
            stack.cv_counter+=1

    # measure performance on both estimation and validation data
    stack.valmode=True
    
    #stack.nests=1

    stack.evaluate(1)
    
    stack.append(nm.mean_square_error)
    
    corridx=nu.find_modules(stack,'correlation')
    if not corridx:
       # add MSE calculator module to stack if not there yet
        stack.append(nm.correlation)
                    
    #print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
                 #stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}".format(stack.meta['mse_est'],
                 stack.meta['mse_val'],stack.meta['r_est'],stack.meta['r_val']))
    valdata=[i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx=valdata[0]
    else:
        stack.plot_dataidx=0
        
        
    # edit: added autoplot kwarg for option to disable auto plotting
    #       -jacob, 6/20/17
    if autoplot:
        stack.quick_plot()
    
    
    # save
    filename=(
            "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl"
            .format(batch, cellid, modelname)
            )
    nu.save_model(stack,filename) 
    #os.chmod(filename, 0o666)

    return(stack)

"""
load_single_model - load and evaluate a model, specified by cellid, batch and modelname

example:
    import lib.nems_main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    stack=nems.load_single_model(cellid,batch,modelname)
    stack.quick_plot()
    
"""
def load_single_model(cellid, batch, modelname):
    
    filename=(
            "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl"
            .format(batch, cellid, modelname)
            )
    # For now don't do anything different to cross validated models.
    # TODO: should these be loaded differently in the future?
    #filename = filename.strip('_xval')
    
    stack=nu.load_model(filename)
    stack.evaluate()
    
    return stack
    




#TODO: Re-work queue functions for use with cluster

# Copy-paste from nems_analysis > __init__.py for db setup

# sets how often sql alchemy attempts to re-establish connection engine
POOL_RECYCLE = 7200;

#create base class to mirror existing database schema
Base = automap_base()
# create a database connection engine

# this points to copy of database, not lab database
# TODO: set up second connection for lab db? jobs added to lab db, but
#       results saved to copy

# get correct path to db info file
libmod_path = os.path.abspath(nm.__file__)
i = libmod_path.find('nems/nems')
nems_path = libmod_path[:i+10]

db = {}
with open (nems_path + "config/hidden/database_info.txt","r") as f:
    for line in f:
        key,val = line.split()
        db[key] = val

SQLALCHEMY_DATABASE_URI = (
        'mysql+pymysql://%s:%s@%s/%s'
        %(db['user'],db['passwd'],db['host'],db['database'])
        )

engine = create_engine(SQLALCHEMY_DATABASE_URI, pool_recycle=POOL_RECYCLE)
Base.prepare(engine, reflect=True)

tQueue = Base.classes.tQueue
NarfResults = Base.classes.NarfResults

# import this when another module needs to use the database connection.
# used like a class - ex: 'session = Session()'
Session = sessionmaker(bind=engine)


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
    for model in modellist:
        for cell in celllist:
            enqueue_single_model(cell, batch, model, force_rerun)
    
    data = 'Placeholder success/failure messsage for user if any.'
    return data


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
    tQueueId = -1
    
    # TODO: anything else needed here? this is syntax for nems_fit_single
    #       command prompt wrapper in main nems folder.
    commandPrompt = (
            "nems_fit_single %s %s %s"
            %(cellid,batch,modelname)
            )

    note = "%s/%s/%s"%(cellid,batch,modelname)
    
    result = (
            session.query(NarfResults)
            .filter(NarfResults.cellid == cellid)
            .filter(NarfResults.batch == batch)
            .filter(NarfResults.modelname == modelname)
            .all()
            )
    if result and not force_rerun:
        print("Entry in NarfResults already exists for: %s, skipping.\n"%note)
        return -1
    
    #query tQueue to check if entry with same cell/batch/model already exists
    qdata = session.query(tQueue).filter(tQueue.note == note).all()
    
    # if it does, check its 'complete' status and take different action based on
    # status
    
    if qdata and (int(qdata[0].complete) <= 0):
        #TODO:
        #incomplete entry for note already exists, skipping
        #update entry with same note? what does this accomplish?
        #moves it back into queue maybe?
        print("Incomplete entry for: %s already exists, skipping.\n"%note)
        return -1
    elif qdata and (int(qdata[0].complete) == 2):
        #TODO:
        #dead queue entry for note exists, resetting
        #update complete and progress status each to 0
        #what does this do? doesn't look like the sql is sent right away,
        #instead gets assigned to [res,r]
        print("Dead queue entry for: %s already exists, resetting.\n"%note)
        qdata[0].complete = 0
        qdata[0].progress = 0
        return -1
    elif qdata and (int(qdata[0].complete) == 1):
        #TODO:
        #resetting existing queue entry for note
        #update complete and progress status each to 0
        #same as above, what does this do?
        print("Resetting existing queue entry for: %s\n"%note)
        qdata[0].complete = 0
        qdata[0].progress = 0
        return -1
    else: #result must not have existed? or status value was greater than 2
        # add new entry
        print("Adding job to queue for: %s\n"%note)
        job = tQueue()
        session.add(add_model_to_queue(commandPrompt,note,job))
    
    # uncomment session.commit() when ready to test saving to database
    session.commit()
    tQueueId = job.id
    session.close()
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

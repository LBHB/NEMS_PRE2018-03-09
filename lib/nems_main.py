#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 18 20:16:37 2017

@author: svd
"""

import lib.nems_modules as nm
import lib.nems_fitters as nf
import lib.nems_utils as nu
import lib.nems_keywords as nk
import lib.baphy_utils as baphy_utils


"""
fit_single_model - create, fit and save a model specified by cellid, batch and modelname

example fit on nice IC cell:
    import lib.nems_main as nems
    cellid='bbl061h-a1'
    batch=291
    modelname='fb18ch100_ev_fir10_dexp_fit00'
    nems.fit_single_model(cellid,batch,modelname)

"""
def fit_single_model(cellid, batch, modelname, autoplot=True):
    
    stack=nm.nems_stack()
    
    stack.meta['batch']=batch
    stack.meta['cellid']=cellid
    stack.meta['modelname']=modelname

    # extract keywords from modelname    
    keywords=modelname.split("_")
    
    # evaluate each keyword in order
    for k in keywords:
        f = getattr(nk, k)
        f(stack)

    # measure performance on both estimation and validation data
    stack.valmode=True
    stack.evaluate(1)
    corridx=nu.find_modules(stack,'correlation')
    if not corridx:
        # add MSE calculator module to stack if not there yet
        stack.append(nm.correlation)
    
    print("Final r_est={0} r_val={1}".format(stack.meta['r_est'],stack.meta['r_val']))
    
    # default results plot, show validation data if exists
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
    filename="/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl".format(batch,cellid,modelname)
    nu.save_model(stack,filename)
    return stack

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
def load_single_model(cellid,batch,modelname):
    
    filename="/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.pkl".format(batch,cellid,modelname)
    stack=nu.load_model(filename)
    stack.evaluate()
    
    return stack
    


#TODO: Re-work queue functions for use with cluster
#TODO: Add in engine, classes & Sessionmaker separate from app for db
#      connection (if want to use sqlalchemy)

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
    for model in modellist:
        for cell in celllist:
            enqueue_single_model(cell,batch,model)
    
    data = 'Placeholder success/failure messsage for user if any.'
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
    
    if result and result['complete'] <= 0:
        #TODO:
        #incomplete entry for note already exists, skipping
        #update entry with same note? what does this accomplish?
        #moves it back into queue maybe?
        pass
    elif result and result['complete'] == 2:
        #TODO:
        #dead queue entry for note exists, resetting
        #update complete and progress status each to 0
        #what does this do? doesn't look like the sql is sent right away,
        #instead gets assigned to [res,r]
        pass
    elif result and result['complete'] == 1:
        #TODO:
        #resetting existing queue entry for note
        #update complete and progress status each to 0
        #same as above, what does this do?
        pass
    else: #result must not have existed? or status value was greater than 2
        # add new entry
        job = tQueue()
        session.add(add_model_to_queue(commandPrompt,note,job))
    
    # uncomment session.commit() when ready to test saving to database
    #session.commit()
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
"""
Created on Jan 5, 2018

@author: svd

This is meant to illustrate the range of uses either in place or that 
would be nice to support
"""

import nems.main as nems
import nems.utilities as nu


# CASE 1
# vanilla NAT model on "nice" linear cell
cellid="chn029d-a1"
batch=271 #A1
modelname="fb18ch100_wcg01_fir15_fit01"

stack=nems.fit_single_model(cellid, batch, modelname, saveInDB=False)



# CASE 2
# pupil-dependent gain - stretch out all trials (rather than averaging across
# stimulus repetitions). that way you can adjust gain according to pupil at 
# each point in time
cellid="BOL006b-48-1"
batch=293 # tone pip (PPS) + pupil
modelname="parm50_wc01_fir10_pupgain_fit01_nested5"

stack=nems.fit_single_model(cellid, batch, modelname, saveInDB=False)


# CASE 3
# pupil-dependent AND task-dependent gain. as in CASE 3, stretched out in
# time. gain on output of STRF is now a function of both pupil and task 
# condition typically would want to add "_nested10" or something on the end.
cellid="TAR010c-30-1"
batch=301 # tone detect (PTD) + pupil + behavior
modelname="parm100pt_wcg02_fir15_stategain_fit01"

stack=nems.fit_single_model(cellid, batch, modelname, saveInDB=False)


# CASE 4
# wacky population STRF. This is mostly SVD's work in progress, but it should run
import nems.poplib as poplib

site='TAR010c16'
batch=271
fmodelname="fchan100_wc02_stp1pc_fir15_dexp_fit01"
factorCount=2
modelname="ssfb18ch100_wc02_stp1pc_fir15"
base_modelname="fb18ch100_wc02_stp1pc_fir15_dexp_fit01"

# generate the model based on previous fits to the output of factor analysis
stack=poplib.pop_factor_strf_init(site=site,factorCount=factorCount,batch=batch,fmodelname=fmodelname,modelname=modelname)

# fit the model
stack=poplib.pop_factor_strf_fit(stack)

# compare performance to the same model fit to each unit in isolation
poplib.pop_factor_strf_eval(stack, base_modelname=base_modelname)





# CASE 5 (NOT IMPLEMENTED YET)
# fit separate STRF for active vs. passive. now we can average across trials
# within a given behavior condition, since we're ignoring pupil
# 
# "fit01wholemodelperbehaviorcondition" could probably have a shorter name
cellid="TAR010c-30-1"
batch=301 # tone detect (PTD) + pupil + behavior
modelname="parm100a_wcg02_fir15_fit01wholemodelperbehaviorcondition"
# this should work:
#modelname="parm100a_wcg02_fir15_fit01"

stack=nems.fit_single_model(cellid, batch, modelname, saveInDB=False)


# CASE 6 (NOT IMPLEMENTED YET)
# generic user scheme. load the data and then ask nems to do the fitting
file=nu.baphy.get_celldb_file(batch, cellid,fs=100, stimfmt='ozgf', chancount=18)
d=nu.io.load_baphy_data(est_files=[file], fs=100)

X=d['stim']
Y=d['resp']

X_est=X[:,3:,:]
Y_est=Y[:,3:,:]
X_val=X[:,3:,:]
Y_val=Y[:,3:,:]

# this is pseudo code, hasn't been written yet!
stack=fit_dumb_model(stim=X_est,resp=Y_est)
r=test_dumb_model(stack=stack,stim=X_val,resp=Y_val)




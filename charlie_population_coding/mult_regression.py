#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:05:17 2017

@author: hellerc
"""

from pop_utils import load_population_stack
import numpy as np
from NRF_tools import NRF_fit, plt_perf_by_trial, eval_fit
import matplotlib.pyplot as plt
import nems.db as ndb
import nems.utilities as ut
batch=301
modelname= "fb18ch100x_wcg02_fir15_dexp_fit01_nested5"
d=ndb.get_batch_cells(batch=301)
cellids=d['cellid']
resp, pred, pupil, a_p = load_population_stack(modelname=modelname,batch=batch)
pre_stim = 0.35
post_stim = 0.35
duration = 0.75
fs = 100
stimulus = np.hstack((np.zeros(int(pre_stim*fs)), np.ones(int(duration*fs)), np.zeros(int(post_stim*fs))))

# Dropping any reps in which there were Nans for one or more stimuli (quick way
# way to deal with nana. This should be improved)

inds = []
for ind in np.argwhere(np.isnan(pupil[0,:,:])):
    inds.append(ind[0])
inds = np.array(inds)
drop_inds=np.unique(inds)
keep_inds=[x for x in np.arange(0,len(resp[0,:,0,0])) if x not in inds]

resp = resp[:,keep_inds,:,:]
pred = pred[:,keep_inds,:,:]
pupil = pupil[:,keep_inds,:]
a_p=np.array(a_p)[keep_inds]
pup = pupil
stim = np.transpose(np.tile(stimulus,[resp.shape[1],resp.shape[2],resp.shape[3],1]),[3, 0, 1, 2])

bincount=resp.shape[0]
repcount=resp.shape[1]
stimcount=resp.shape[2]
cellcount=resp.shape[3]

a_p_unwrapped = list(np.tile(a_p, (30,1)).T.reshape(repcount*stimcount))
pupil_unwrapped = np.nanmean(pupil,0).reshape(repcount*stimcount)
rN = NRF_fit(r=resp,r0_strf=pred,model='NRF_STRF',spontonly=False)
fig = plt_perf_by_trial(resp,rN,pred,combine_stim=False,a_p=a_p_unwrapped,pupil=pupil,pop_state={'method': 'SVD', 'dims':5})

# Evaluate r0 and rN
cc_rN = eval_fit(resp,rN)
cc_r0 = eval_fit(resp,pred)

cc_rN_flat = cc_rN['bytrial'].reshape(repcount*stimcount,cellcount)
cc_r0_flat = cc_r0['bytrial'].reshape(repcount*stimcount,cellcount)

import statsmodels.api as sm
import pandas as pd

reg_in = pd.DataFrame(np.vstack((a_p_unwrapped,pupil_unwrapped)).T, columns = ['behavior', 'pupil'])
reg_out = pd.DataFrame(cc_rN_flat-cc_r0_flat, columns = cellids)
for cid in reg_out:
    if (np.any(np.isnan(reg_out[cid]))):
        for i, val in enumerate(reg_out[cid]):
            if np.isnan(val):
                reg_out[cid][i]=0
                
#optional- drop columns with nans
for cid in reg_out:
    if (np.any(np.isnan(reg_out[cid]))):      
        reg_out = reg_out.drop(cid,axis=1)
        
cellids=reg_out.columns.values
X = reg_in[['behavior','pupil']]
X = sm.add_constant(X)
b = reg_in['behavior']
b = sm.add_constant(b)
p= reg_in['pupil']
p = sm.add_constant(p)
p_coef = []
b_coef=[]
rsq_full=[]
rsq_b = []
rsq_p=[]
aic_p=[]
aic_b=[]
aic_full=[]
for cid in reg_out:
    y = reg_out[cid] 
    fullmodel=sm.OLS(y,X).fit()
    pModel=sm.OLS(y,p).fit()
    bModel=sm.OLS(y,b).fit()
    p_coef.append(fullmodel.params['pupil'])
    b_coef.append(fullmodel.params['behavior'])
    rsq_full.append(fullmodel.rsquared)
    rsq_p.append(pModel.rsquared)
    rsq_b.append(bModel.rsquared)
    aic_p.append(pModel.bic)
    aic_full.append(fullmodel.bic)
    aic_b.append(bModel.bic)

aic_p = np.array(aic_p)
aic_b = np.array(aic_b)
aic_full = np.array(aic_full)

fig = plt.figure()
LINE_STYLES=['solid','dashed','dashdot','dotted']
NUM_STYLES=4
cm = plt.get_cmap('gist_rainbow')
ax = fig.add_subplot(121)
ax.set_color_cycle([cm(1.*i/len(cellids)) for i in range(len(cellids))])
plt.title('R^2')
for i in range(0, len(cellids)):
    lines = plt.plot(np.vstack((rsq_b,rsq_p,rsq_full))[:,i],'.-')
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
plt.plot(np.mean(np.vstack((rsq_b,rsq_p,rsq_full)),1),'.-k',lw=3)
plt.xticks(range(0,3),['Behavior','Pupil','Both'])
ax = fig.add_subplot(122)
ax.set_color_cycle([cm(1.*i/len(cellids)) for i in range(len(cellids))])
plt.title('AIC')
for i in range(0,len(cellids)):
    lines=plt.plot(np.vstack((aic_b-aic_p,aic_full-aic_b,aic_p-aic_b,aic_full-aic_p))[:,i],'.-',markersize=9)
    lines[0].set_linestyle(LINE_STYLES[i%NUM_STYLES])
plt.plot(np.mean(np.vstack((aic_b-aic_p,aic_full-aic_b,aic_p-aic_b,aic_full-aic_p)),1),'.-k',lw=3)
plt.axhline(0,linestyle='--',color='gray',lw=4)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.99, box.height])
plt.xticks(range(0,4),['Behavior-Pupil','Both - Behavior','Pupil-Behavior','Both - Pupil'])
ax.legend(cellids,loc='center left', bbox_to_anchor=(1, 0.5))

 


b_cells_rsq = np.vstack((rsq_b,rsq_p,rsq_full))[:,np.argwhere(aic_b < aic_p)].squeeze()
p_cells_rsq = np.vstack((rsq_b,rsq_p,rsq_full))[:,np.argwhere(aic_p < aic_b)].squeeze()
indsb = []
for i in range(0,len(np.argwhere(aic_b < aic_p))):
    indsb.append(np.argwhere(aic_b < aic_p)[i][0])
indsp=[]
for i in range(0,len(np.argwhere(aic_p < aic_b))):
    indsp.append(np.argwhere(aic_p < aic_b)[i][0])
b_cells_aic = np.vstack((aic_b-aic_p,aic_full-aic_b))[:,np.argwhere(aic_b < aic_p)].squeeze()
p_cells_aic = np.vstack((aic_p-aic_b,aic_full-aic_p))[:,np.argwhere(aic_p< aic_b)].squeeze()
   
fig = plt.figure()    
ax = fig.add_subplot(221)
plt.title('R^2')
for i in range(0, b_cells_rsq.shape[1]):
    lines = plt.plot(b_cells_rsq[:,i],'.-')
plt.plot(np.mean(b_cells_rsq,1),'.-k',lw=3)
plt.xticks(range(0,3),['Behavior','Pupil','Both'])

ax = fig.add_subplot(222)
plt.title('AIC')
for i in range(0,(b_cells_aic.shape[1])):
    lines=plt.plot(b_cells_aic[:,i],'.-',markersize=9)
plt.plot(np.mean(b_cells_aic,1),'.-k',lw=3)
plt.axhline(0,linestyle='--',color='gray',lw=4)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.99, box.height])
ax.legend(cellids[indsb],loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(range(0,2),['Behavior-Pupil','Both - Behavior'])


ax = fig.add_subplot(223)
plt.title('R^2')
for i in range(0,p_cells_rsq.shape[1]):
    lines=plt.plot(p_cells_rsq[:,i],'.-',markersize=9)
plt.plot(np.mean(p_cells_rsq,1),'.-k',lw=3)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.99, box.height])
plt.xticks(range(0,3),['Behavior', 'Pupil','Both - Pupil'])  
   
ax = fig.add_subplot(224)
plt.title('AIC')
for i in range(0,p_cells_aic.shape[1]):
    lines=plt.plot(p_cells_aic[:,i],'.-',markersize=9)
plt.plot(np.mean(p_cells_aic,1),'.-k',lw=3)
plt.axhline(0,linestyle='--',color='gray',lw=4)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.99, box.height])
ax.legend(cellids[indsp].squeeze(),loc='center left', bbox_to_anchor=(1, 0.5))
plt.xticks(range(0,2),['Pupil-Behavior','Both - Pupil'])    

from pop_utils import whiten
resp_PCA = whiten(resp)
resp_PCA = resp_PCA.reshape(bincount*repcount*stimcount, cellcount)
U,S,V = np.linalg.svd(resp_PCA,full_matrices=False)
S = S**2 # converting to variance explained from sigma
ndims = 5

PCs = np.mean(U[:,0:ndims].reshape(bincount, repcount, stimcount, ndims),0).reshape(repcount*stimcount,ndims)

p_coef = []
b_coef=[]
rsq_full=[]
rsq_b = []
rsq_p=[]
aic_p=[]
aic_b=[]
aic_full=[]
for i in range(0,ndims):
    reg_out=pd.DataFrame(PCs[:,i],columns=['pc'+str(i)])
    y = reg_out['pc'+str(i)] 
    fullmodel=sm.OLS(y,X).fit()
    pModel=sm.OLS(y,p).fit()
    bModel=sm.OLS(y,b).fit()
    p_coef.append(fullmodel.params['pupil'])
    b_coef.append(fullmodel.params['behavior'])
    rsq_full.append(fullmodel.rsquared)
    rsq_p.append(pModel.rsquared)
    rsq_b.append(bModel.rsquared)
    aic_p.append(pModel.bic)
    aic_full.append(fullmodel.bic)
    aic_b.append(bModel.bic)

aic_p = np.array(aic_p)
aic_b = np.array(aic_b)
aic_full = np.array(aic_full)

plt.figure()
plt.subplot(121)
plt.plot(np.vstack((rsq_b,rsq_p,rsq_full)),'.-')
plt.xticks(range(0,3),['behavior','pupil','both'])
plt.legend(['pc1','pc2','pc3','pc4','pc5'])
plt.subplot(122)
plt.plot(np.vstack((aic_b-aic_full,aic_p-aic_full)),'.-')
plt.xticks(range(0,2),['b-full','p-full'])
plt.legend(['pc1','pc2','pc3','pc4','pc5'])

cc_rN_active = np.nanmean(cc_rN['bytrial'][a_p==1,:,:],-1).flatten()
cc_rN_passive = np.nanmean(cc_rN['bytrial'][a_p==0,:,:],-1).flatten()
cc_r0_active = np.nanmean(cc_r0['bytrial'][a_p==1,:,:],-1).flatten()
cc_r0_passive = np.nanmean(cc_r0['bytrial'][a_p==0,:,:],-1).flatten()

plt.figure()
plt.subplot(121)
plt.title('Passive')
plt.plot(cc_r0_passive,cc_rN_passive,'.')
plt.plot(np.arange(0.05,0.35,0.01),np.arange(0.05,0.35,0.01))
plt.xlim((0.05, 0.35))
plt.ylim((0.05, 0.4))
plt.subplot(122)
plt.title('active')
plt.plot(cc_r0_active,cc_rN_active,'.')
plt.plot(np.arange(0.05,0.35,0.01),np.arange(0.05,0.35,0.01))
plt.xlim((0.05, 0.35))
plt.ylim((0.05, 0.4))

cc_rN_active = np.nanmean(np.nanmean(cc_rN['bytrial'][a_p==1,:,:],0),0).flatten()
cc_rN_passive = np.nanmean(np.nanmean(cc_rN['bytrial'][a_p==0,:,:],0),0).flatten()
cc_r0_active = np.nanmean(np.nanmean(cc_r0['bytrial'][a_p==1,:,:],0),0).flatten()
cc_r0_passive = np.nanmean(np.nanmean(cc_r0['bytrial'][a_p==0,:,:],0),0).flatten()

fig=plt.figure()
ax = fig.add_subplot(121)
ax.set_color_cycle([cm(1.*i/len(cellids)) for i in range(len(cellids))])
plt.title('Passive')
for i in range(0,len(cellids)):
     plt.plot(cc_r0_passive[i],cc_rN_passive[i],'.')
plt.plot(np.arange(0,0.6,0.01),np.arange(0,.6,0.01))
plt.xlabel('null model')
plt.ylabel('network model')
ax=fig.add_subplot(122)
ax.set_color_cycle([cm(1.*i/len(cellids)) for i in range(len(cellids))])
plt.title('active')
for i in range(0, len(cellids)):
     plt.plot(cc_r0_active[i],cc_rN_active[i],'.')
plt.plot(np.arange(0,0.6,0.01),np.arange(0,0.6,0.01))
plt.xlabel('null model')
plt.ylabel('network model')

from scipy.ndimage.filters import gaussian_filter1d
plt.figure()
plt.subplot(521)
plt.title('Pupil')
plt.plot(gaussian_filter1d(pupil_unwrapped,5))
plt.subplot(522)
plt.title('model')
plt.plot(gaussian_filter1d(np.nanmean(cc_rN['bytrial'],-1).reshape(11*30)-np.nanmean(cc_r0['bytrial'],-1).reshape(11*30),5))
plt.subplot(523)
plt.title('pc1')
plt.plot(gaussian_filter1d(PCs[:,0],5))
plt.subplot(524)
plt.title('pc2')
plt.plot(gaussian_filter1d(PCs[:,1],5))
plt.subplot(525)
plt.title('pc3')
plt.plot(gaussian_filter1d(PCs[:,2],5))
plt.subplot(526)
plt.title('pc4')
plt.plot(gaussian_filter1d(PCs[:,4],5))
plt.subplot(527)
plt.title('pc5')
plt.plot(gaussian_filter1d(PCs[:,4],5))
plt.subplot(528)
plt.plot(gaussian_filter1d(np.nanmean(cc_rN['bytrial'],-1).reshape(11*30),5),'r')
plt.plot(gaussian_filter1d(np.nanmean(cc_r0['bytrial'],-1).reshape(11*30),5),'b')
plt.legend(['rN','r0'])
plt.subplot(529)
plt.plot(a_p_unwrapped)
plt.title('behavior')
plt.tight_layout()


plt.figure()
plt.bar(0,np.corrcoef(np.nanmean(cc_rN['bytrial'],-1).reshape(11*30), pupil_unwrapped)[0][1])
plt.bar(1,np.corrcoef(np.nanmean(cc_r0['bytrial'],-1).reshape(11*30), pupil_unwrapped)[0][1])
plt.bar(2,abs(np.corrcoef(np.nanmean(cc_rN['bytrial'],-1).reshape(11*30)-np.nanmean(cc_r0['bytrial'],-1).reshape(11*30), pupil_unwrapped)[0][1]))



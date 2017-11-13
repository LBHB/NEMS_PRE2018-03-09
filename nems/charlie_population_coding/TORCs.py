#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 17:31:05 2017

@author: hellerc
"""

# Network model analysis of Tartufo TORC data w/ STRF as null model

# ------------------------ Load Data ------------------------------------
from importlib import reload
import nems.db as ndb
import nems.utilities as ut
import nems.stack as ns

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


reload(ut)

d=ndb.get_batch_cells(batch=301)
cellid=d['cellid'][1]  # or cycle through all entries to list them all...
'''
filename=ut.baphy.get_celldb_file(301,cellid,fs=100,stimfmt="ozgf",chancount=18)
data=[]
for ll in range(0,4):
    data.append(ut.baphy.load_baphy_file(filename,level=ll))
'''    
# If need to query db
user = 'david'
passwd = 'nine1997'
host = 'hyrax.ohsu.edu'
database = 'cell'


# Load models
cellids = d['cellid'][0:]
rvals=[]
stacks=[]

batch=301
modelname= "fb18ch100x_wcg02_fir15_dexp_fit01_nested5"
        
import sys
a_p = []   #active or passive repetition -- 1 is active, 0 is passive
#length= int(len(data))*2
for j in np.arange(1, 8,2):
    
    for i, cellid in enumerate(cellids):
        if j == 1:
            stack=ut.io.load_single_model(cellid, batch, modelname)
            stacks.append(stack)
        
        p = stacks[i].data[-1][j]['pred'].copy()
        r = stacks[i].data[-1][j]['resp_raw'].copy()
        p = np.transpose(np.tile(p, (r.shape[1],1,1)).T, (0, 2, 1))
        pup = np.transpose(stacks[i].data[-1][j]['pupil'].copy(), (2,0,1))
        
        if i == 0:
            if stack.data[-1][j]['stimparam'][0].find('_a_') > 0:
                for z in range(0, stack.data[-1][j]['resp_raw'].shape[1]):
                    a_p.append(1)
            else:
                for z in range(0, stack.data[-1][j]['resp_raw'].shape[1]):
                    a_p.append(0)
    
        if j == 1:
            if i == 0:
                pred = np.empty((r.shape[0], r.shape[1], r.shape[2], len(cellids)))
                resp = np.empty((r.shape[0], r.shape[1], r.shape[2], len(cellids)))
                pupil = np.empty((r.shape[0],r.shape[1],r.shape[2]))
                ptemp = np.empty((r.shape[0], r.shape[1], r.shape[2], len(cellids)))
                rtemp = np.empty((r.shape[0], r.shape[1], r.shape[2], len(cellids)))
                pupTemp = np.empty((r.shape[0], r.shape[1], r.shape[2]))
            
            pred[:,:,:,i]=p
            resp[:,:,:,i]=r
            pupil[:,:,:]=pup
            
            ptemp[:,:,:,i]=p
            rtemp[:,:,:,i]=r
            pupTemp[:,:,:]=pup
        
        if j > 1:
            if i == 0:
                ptemp = np.empty((r.shape[0], r.shape[1], r.shape[2], len(cellids)))
                rtemp = np.empty((r.shape[0], r.shape[1], r.shape[2], len(cellids)))
                pupTemp = np.empty((r.shape[0], r.shape[1], r.shape[2]))
               
            ptemp[:,:,:,i]=p
            rtemp[:,:,:,i]=r
            pupTemp[:,:,:]=pup

    if j > 1:
        pred = np.concatenate((pred, ptemp),axis=1)
        resp = np.concatenate((resp, rtemp),axis=1)
        pupil = np.concatenate((pupil, pupTemp),axis=1)
        
    r_val=stack.meta['r_val']  # test set prediction accuracy
    rvals.append(r_val)
                

# Dropping any reps in which there were Nans for one or more stimuli (quick way
# way to deal with nana. This should be improved)

inds = []
for ind in np.argwhere(np.isnan(pupil[0,:,:])):
    inds.append(ind[0])
inds = np.array(inds)
drop_inds=np.unique(inds)
keep_inds=[x for x in np.arange(0,len(resp[0,:,0,0])) if x not in inds]

a_p = np.array(a_p)[keep_inds]
resp = resp[:,keep_inds,:,:]
pred = pred[:,keep_inds,:,:]
pupil = pupil[:,keep_inds,:]
pup = pupil

from NRF_tools import NRF_fit, eval_fit
rN = NRF_fit(resp, r0_strf=pred, model="NRF_STRF", spontonly=0, shuffle=True)
rN_perf = eval_fit(resp, rN)
r0_perf = eval_fit(resp, pred)


cc_rN_all = np.empty((resp.shape[1], resp.shape[2], resp.shape[-1]))
cc_r0_all = np.empty((resp.shape[1], resp.shape[2], resp.shape[-1]))
for i in range(0, resp.shape[1]):
    for stim in range(0, resp.shape[2]):
        for cell in range(0, resp.shape[-1]):
            cc_rN_all[i, stim, cell]=np.corrcoef(rN[:,i, stim,cell], resp[:,i,stim,cell])[0][1]
            cc_r0_all[i,stim, cell]=np.corrcoef(pred[:,i, stim,cell], resp[:,i,stim,cell])[0][1]
            
cc_rN = np.mean(cc_rN_all[:,:,:],1)
cc_r0 = np.mean(cc_r0_all[:,:,:],1)
pup = np.mean(pup[:,:,:],2)

act_pass = list(a_p)
for i, x in enumerate(act_pass):
    if x == 1:
        act_pass[i] = 'green'
    elif x == 0:
        act_pass[i] = 'yellow'
plt.figure(1)
plt.subplot2grid((3,3),(0,0),colspan=2)
plt.plot(np.nanmean(cc_rN,1), '-', color='r', alpha = 0.3)
plt.scatter(np.arange(0,len(cc_rN)), np.nanmean(cc_rN, 1), c=act_pass)
plt.plot(np.nanmean(cc_r0,1), '-', color='b', alpha = 0.3)
plt.scatter(np.arange(0,len(cc_rN)), np.nanmean(cc_r0, 1), c=act_pass)
plt.ylabel('pearsons corr coef')
plt.legend(['rN', 'r0 (strf)'], loc = 'upper right', fontsize = 'large')
#plt.xlabel('pip trial')
pup_m = np.mean(pup,0)/2
plt.title('rN vs. pupil: %s, r0 vs. pupil: %s' 
          %(np.corrcoef(np.nanmean(cc_rN,1), pup_m)[0][1], np.corrcoef(np.nanmean(cc_r0,1), pup_m)[0][1]))

diff = np.nanmean(cc_rN,1)-np.nanmean(cc_r0,1)
plt.subplot2grid((3,3),(1,0),colspan=2)
plt.plot(pup_m/2,'-o', color='k', alpha=0.5, lw=2)
plt.scatter(np.arange(0,len(cc_rN)), diff, c=act_pass)
plt.plot(diff, '-', color='g', alpha = 0.3)
plt.legend(['pup', 'rN-r0'])
#plt.xlabel('pip trials')
plt.title('corr coef btwn rN-r0 and pupil: %s' 
          %(np.corrcoef(diff, pup_m)[0][1]))
        

plt.figure(1)
# Trying PCA - how does it related with the network model performance
fs = 10
import scipy.signal as ss
samps = int(round(fs*(resp.shape[0]/100.0)))
resp_PCA = ss.resample(resp, samps)

bincount_pca = resp_PCA.shape[0]
repcount = resp_PCA.shape[1]
stimcount = resp_PCA.shape[2]
cellcount = resp_PCA.shape[3]
import pop_utils as pu
resp_PCA = pu.whiten(resp_PCA)
resp_PCA = np.reshape(resp_PCA, (bincount_pca*repcount*stimcount, cellcount))
U,S,V = np.linalg.svd(resp_PCA)

plt.subplot2grid((3,3),(2,0),colspan=2)
for i in range(0,5):
    pc = np.mean(np.mean(U[:,i].reshape(bincount_pca, repcount, stimcount),0),1)
    plt.plot(pc, lw = 2, label = 'pc: %s ,pupil: %s ,model: %s' %((i+1),round(np.corrcoef(pup_m, pc)[0][1],2),round(np.corrcoef(diff, pc)[0][1],2)))
    plt.legend()

plt.subplot2grid((3,3), (0, 2), colspan=1)
for i in range(0, len(S)):
    plt.bar(i, 100*(sum(S[0:i])/sum(S)), color='b')
    plt.ylabel('percent variance explained')
    plt.xlabel('number of pcs')

plt.subplot2grid((3,3),(1,2),colspan=1)
for i in range(0, len(S)):
    plt.bar(i, 100*(S[i]/sum(S)), color='b')
    plt.ylabel('percent variance explained')
    plt.xlabel('Inidividual pc')
    
plt.subplot2grid((3,3),(2,2),colspan=1)
fs = 100
plt.scatter(np.arange(0, 11), fs*np.mean(np.mean(np.mean(resp,0),1),1), c=act_pass)
plt.plot(fs*np.mean(np.mean(np.mean(resp,0),1),1), 'r', alpha = 0.7)
plt.ylabel('Hz')
plt.xlabel('Repetition')
plt.tight_layout()


print('Comparing different models...')
def onpick3(event):
    ind = event.ind
    for i in ind:
        print('onpick3 scatter:', cellids[i])

x = np.linspace(-1,1,3)
ncols = 3
nrows = 4
cellcount = resp.shape[-1]
repcount = len(keep_inds)
bincount = resp.shape[0]
spontonly=0
fs=100
stim=0
color = np.arange(1,cellcount+1)
fig = plt.figure()
for rep in range(0, repcount):
    ax = fig.add_subplot(nrows,ncols,rep+1)
    ax.scatter(np.nanmean(r0_perf['bytrial'][rep,:,:],0), np.nanmean(rN_perf['bytrial'][rep,:,:],0), c=color,s=10,picker=True)
    ax.plot(x, x, '-k',lw=2)
    ax.axis([-.1,1,-.1,1])
    ax.set_title(rep+1, fontsize=7)
    fig.canvas.mpl_connect('pick_event',onpick3)
    if rep != (ncols*nrows - ncols):
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        ax.set_ylabel('rN')
        ax.set_xlabel('r0')
rawid = 'TAR'
fig.suptitle('r0 vs. rN for each cell \n stim: %s, rawid: %s, spontonly: %s, fs: %s' %(stim+1, rawid, spontonly, fs))


# ===== Split up performance within each repetition based on pupil size ======
npcs = 5
pupil_m = np.mean(pupil,0)
p_sorted = np.empty((pupil_m.shape[0]*2))
p_medians = np.empty((len(keep_inds)))
cc_rN_sorted = np.empty((cc_rN.shape[0]*2))
cc_r0_sorted = np.empty((cc_r0.shape[0]*2))
pcs_sorted = np.empty((npcs, cc_rN.shape[0]*2))
it = 0
for i in range(0, len(keep_inds)):
    p_medians[i] = np.median(pupil_m[i,:])
    big_inds = np.argwhere(pupil_m[i, :] > p_medians[i])
    big_inds =  [int(x) for x in big_inds]
    little_inds = np.argwhere(pupil_m[i,:] <= p_medians[i])
    little_inds =  [int(x) for x in little_inds]
    
    for j in range(0,npcs):
        pcs_sorted[j, it] = np.mean(U[:,j].reshape(bincount_pca, repcount, stimcount)[:,i,big_inds])
        pcs_sorted[j, it+1] = np.mean(U[:,j].reshape(bincount_pca, repcount, stimcount)[:,i,big_inds])
        
    cc_rN_sorted[it] = np.nanmean(cc_rN_all[i,big_inds,:])
    cc_rN_sorted[it+1] =np.nanmean(cc_rN_all[i,little_inds,:])
    cc_r0_sorted[it] = np.nanmean(cc_r0_all[i,big_inds,:])
    cc_r0_sorted[it+1]=np.nanmean(cc_r0_all[i,little_inds,:])
    p_sorted[it] = np.mean(pupil_m[i,big_inds])
    p_sorted[it+1] = np.mean(pupil_m[i,little_inds])
    it+=2
    
act_pass_sorted = [val for pair in zip(act_pass, act_pass) for val in pair]
plt.figure(2)
plt.subplot2grid((3,3),(0,0),colspan=2)
plt.plot(cc_rN_sorted, '-', color='r', alpha = 0.3)
plt.scatter(np.arange(0,len(cc_rN_sorted)), cc_rN_sorted, c = act_pass_sorted)
plt.plot(cc_r0_sorted, '-', color='b', alpha = 0.3)
plt.scatter(np.arange(0,len(cc_rN_sorted)), cc_r0_sorted, c = act_pass_sorted)
plt.ylabel('pearsons corr coef')
plt.xlabel('repetition')
plt.legend(['rN', 'r0 (strf)'])
plt.title('rN vs. pupil: %s, r0 vs. pupil: %s' 
          %(np.corrcoef(cc_rN_sorted, p_sorted)[0][1], np.corrcoef(cc_r0_sorted, p_sorted)[0][1]))

diff = cc_rN_sorted-cc_r0_sorted
plt.subplot2grid((3,3),(1,0),colspan=2)
plt.plot(p_sorted/5,'-o', color='k', alpha=0.5, lw=2)
plt.scatter(np.arange(0,len(cc_rN_sorted)), diff, c=act_pass_sorted)
plt.plot(diff, '-', color='g', alpha = 0.3)
plt.legend(['pup', 'rN-r0'])
plt.xlabel('repetition')
plt.title('corr coef btwn rN-r0 and pupil: %s' 
          %(np.corrcoef(diff, p_sorted)[0][1]))          
            
plt.subplot2grid((3,3),(2,0),colspan=2)
for i in range(0,5):
    plt.plot(pcs_sorted[i,:], lw = 2, label = 'pc: %s ,pupil: %s ,model: %s' %((i+1),round(np.corrcoef(p_sorted, pcs_sorted[i,:])[0][1],2),round(np.corrcoef(diff, pcs_sorted[i,:])[0][1],2)))
    plt.legend()

plt.subplot2grid((3,3), (0, 2), colspan=1)
for i in range(0, len(S)):
    plt.bar(i, 100*(sum(S[0:i])/sum(S)), color='b')
    plt.ylabel('percent variance explained')
    plt.xlabel('number of pcs')

plt.subplot2grid((3,3),(1,2),colspan=1)
for i in range(0, len(S)):
    plt.bar(i, 100*(S[i]/sum(S)), color='b')
    plt.ylabel('percent variance explained')
    plt.xlabel('Inidividual pc')

plt.tight_layout()          


# ========= Unwrap TORCs to look at performance across all stims =============            
act_pass_unwrapped = list(np.tile(a_p, (30,1)).T.reshape(repcount*stimcount))
for i, x in enumerate(act_pass_unwrapped):
    if x == 1:
        act_pass_unwrapped[i] = 'green'
    elif x == 0:
        act_pass_unwrapped[i] = 'yellow'          

act = np.argwhere(np.array(act_pass_unwrapped) == 'green')
s_inds = []
for i in range(0, len(act)-1):
    if act[i+1] != act[0]+act[i]+1:
        s_inds.append(i)            
cc_rN_unwrapped = np.nanmean(cc_rN_all,2).reshape(repcount*stimcount)
cc_r0_unwrapped = np.nanmean(cc_r0_all,2).reshape(repcount*stimcount)        
pupil_unwrapped = np.nanmean(pupil, 0).reshape(repcount*stimcount)

plt.figure(3)
plt.subplot2grid((3,3), (0,0), colspan=2)
plt.plot(cc_rN_unwrapped, 'r', alpha=0.8)
plt.plot(cc_r0_unwrapped, 'b', alpha= 0.8)
plt.axvspan(act[0],act[s_inds[0]],color='k', alpha=0.2)
plt.axvspan(act[s_inds[0]+1],act[-1],color='k', alpha=0.2)
plt.legend(['rN', 'r0 (strf)'], loc = 'upper right', fontsize = 'large')

plt.subplot2grid((3,3), (1,0), colspan=2)
diff = cc_rN_unwrapped - cc_r0_unwrapped
plt.title('model vs. pupil: %s' %(np.corrcoef(pupil_unwrapped, diff)[0][1]))
plt.plot(pupil_unwrapped/2, 'k', alpha=0.8)
plt.plot(diff, 'g', alpha = 0.8)
plt.axvspan(act[0],act[s_inds[0]],color='k', alpha=0.2)
plt.axvspan(act[s_inds[0]+1],act[-1],color='k', alpha=0.2)
plt.legend(['pupil', 'model'], loc = 'upper right', fontsize = 'large')

plt.subplot2grid((3,3), (2,0), colspan=2)
for i in range(0, npcs):
    pc = np.mean(U[:,i].reshape(bincount_pca, repcount, stimcount),0).reshape(repcount*stimcount)
    plt.plot(pc, label = 'pc: %s ,pupil: %s ,model: %s' %((i+1),round(np.corrcoef(pupil_unwrapped, pc)[0][1],2),round(np.corrcoef(diff, pc)[0][1],2)))
plt.legend(loc='upper right')

# ============= Comparing model weights with PC weights =======================
from NRF_tools import get_weight_mat

model_weights = get_weight_mat(resp)
mw = np.empty((len(model_weights), len(model_weights)))
for i in range(0, len(model_weights)):
    mw[i, 0:i] = model_weights[i, 0:i]
    mw[i, i]=np.nan
    mw[i,(i+1):] = model_weights[i,i:]

    

PCA_weights = V[0:10,:]
c_ids = [x[8:] for x in cellids]

plt.figure()
plt.subplot(121)
plt.imshow(mw)
plt.xlabel('neurons')
plt.ylabel('neurons')
plt.xticks(np.arange(0,len(c_ids)), np.array((c_ids)), rotation=90, fontsize=8)
plt.yticks(np.arange(0,len(c_ids)), np.array((c_ids)), fontsize=8)
plt.title('network coupling weights')
plt.subplot(122)
plt.imshow(PCA_weights.T, aspect = 'equal')
plt.xlabel('PCs')
plt.ylabel('Neurons')
plt.title('PC weights')


plt.figure()
for i in range(0,4):
    plt.subplot(2,2,i+1)
    inds = [int(x) for x in list(np.argsort(-V[i,:]))]
    sort1 = mw[inds, :]
    plt.imshow(sort1[:, inds])
    plt.yticks(np.arange(0,len(cellids)), np.array((c_ids))[inds], fontsize=8)
    plt.title('sorted by pc: %s' %(i+1), fontsize=10)
    plt.tick_params(bottom='off', labelbottom='off')


# === Use K-means clustering to group neurons by coupling within the network ===
from sklearn.cluster import KMeans
import numpy as np

nclusters = 4 ## trial and error...

# get rid of nans in coupling matrix for the sake of clustering
mw_0 = mw.flatten()
for i, x in enumerate(mw_0):
    if np.isnan(x):
        mw_0[i] = 0
mw_0 = mw_0.reshape(mw.shape)
kmeans = KMeans(n_clusters=nclusters, random_state=0).fit(mw_0)

# Look at clusters in PCA space
U_, S_, V_ = np.linalg.svd(mw_0)
plt.figure()
plt.scatter(U_[:,0], U_[:,1], c=kmeans.labels_)
plt.legend(['1','2','3','4'])
plt.xlabel('PC1 - weights')
plt.ylabel('PC2 - weights')
plt.title('kmeans clustering')

# Plot PSTHs for each cluster of neurons
resp_ds = ss.resample(resp*fs, samps)   # multiply by fs to get to hz
act = np.argwhere(a_p==1)
s_inds = []
for i in range(0, len(act)-1):
    if act[i+1] != act[0]+act[i]+1:
        s_inds.append(i)  
bincount_ds = resp_ds.shape[0]
plt.figure(4)
for i in range(0, nclusters):
    plt.subplot2grid((nclusters, 3), (i, 0), colspan=2)
    plt.title('cluster: %s' %(i+1))
    for j in np.argwhere(kmeans.labels_ == i):
        plt.plot(np.mean(resp_ds[:,:,:, j],2).T.flatten(), 'b', alpha =0.3)
    plt.plot(np.squeeze(np.mean(np.mean(resp_ds[:,:,:,np.argwhere(kmeans.labels_ == i)], 2),2)).T.flatten(), 'k', lw=4)
    plt.axvspan(act[0],act[s_inds[0]]*bincount_ds+bincount_ds,color='k', alpha=0.2)
    plt.axvspan(act[s_inds[0]+1]*bincount_ds,act[-1]*bincount_ds+bincount_ds,color='k', alpha=0.2)
    plt.ylabel('rate - Hz')
    plt.xlabel('time')
plt.figure(4)
for i in range(0, nclusters):
    plt.subplot2grid((nclusters, 3), (i,2), colspan=1)
    if len(np.argwhere(kmeans.labels_ == i))>1:
        w = np.squeeze(PCA_weights[:,np.argwhere(kmeans.labels_ == i)])
        plt.imshow(w.T, aspect='auto')
        plt.yticks(range(0, len(np.argwhere(kmeans.labels_ == i))),np.array(c_ids)[np.argwhere(kmeans.labels_ == i)], fontsize=6)
    else:
        plt.plot(np.squeeze(PCA_weights[:,np.argwhere(kmeans.labels_ == i)]))
        plt.ylabel(np.array(c_ids)[np.argwhere(kmeans.labels_ == i)])
    plt.xlabel('PCs')
    
# ================ Model performance by cluster of neurons ====================
cc_rN_clust = cc_rN_all.reshape(cc_rN_all.shape[0]*cc_rN_all.shape[1], cc_rN_all.shape[-1])
cc_r0_clust = cc_r0_all.reshape(cc_r0_all.shape[0]*cc_r0_all.shape[1], cc_r0_all.shape[-1])
diff_clust = np.empty((nclusters, cc_rN_all.shape[0]*cc_rN_all.shape[1]))
plt.figure()
for i in range(0, nclusters):
    bar_width = 0.35
    diff_clust[i,:] = np.nanmean(np.squeeze(cc_rN_clust[:, np.argwhere(kmeans.labels_ == i)]- cc_r0_clust[:, np.argwhere(kmeans.labels_ == i)]), 1)
    plt.bar(i, np.mean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)]), bar_width, color='r')
    sem = np.std((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)]))   #/np.sqrt(len((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)])))
    sem2 = np.std((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)]))  #/np.sqrt(len((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)])))
    plt.bar(i+bar_width, np.mean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)]), bar_width, color='b')
    plt.legend(['active', 'passive'])
    plt.errorbar(i, np.mean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)]), yerr = sem, color='k')
    plt.errorbar(i+bar_width, np.mean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)]), yerr = sem2, color='k')
    
plt.xticks(range(0, nclusters), range(1,nclusters+1))
plt.xlabel('neuron clusters')
plt.ylabel('rN pred - r0 prediction')

# by each neuron
plt.figure()
diff_clust = np.empty((cellcount, cc_rN_all.shape[0]*cc_rN_all.shape[1]))
for i in range(0, cellcount):
    bar_width = 0.35
    diff_clust[i,:] = np.squeeze(cc_rN_clust[:, i]- cc_r0_clust[:, i])
    plt.bar(i, np.nanmean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)]), bar_width, color='r')
    sem = np.nanstd((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)]))/np.sqrt(len((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)])))
    sem2 = np.nanstd((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)]))/np.sqrt(len((diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)])))
    plt.bar(i+bar_width, np.nanmean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)]), bar_width, color='b')
    plt.legend(['active', 'passive'])
    plt.errorbar(i, np.nanmean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==1)]), yerr = sem, color='k')
    plt.errorbar(i+bar_width, np.nanmean(diff_clust[i,np.argwhere(np.repeat(a_p, [stimcount])==0)]), yerr = sem2, color='k')
    
plt.xticks(range(0, cellcount), c_ids, rotation = 90, fontsize=8)
plt.xlabel('neuron')
plt.ylabel('rN pred - r0 prediction')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:40 2018

@author: svd
"""

import os
import io
import re
import numpy as np
import scipy.io
import scipy.stats
import scipy.signal

#import nems.recording as Recording
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf

import nems.utilities.baphy
import nems.signal
import nems.recording
import nems.db as nd

def state_gain_model(recording,options):
    """
    fit a model that scales the PSTH by various state variables. general
    purpose version of ptd_gain_model
    
    TODO-allow options to specify list of variables to use for fitting
    """
    resp=recording.get_signal('resp')
    cellid=resp.meta['cellid']
    batch=resp.meta['batch']
   
    SHOW_PLOT=options.get('plot_results',True)
    options['plot_ax']=options.get('plot_ax',None)
    
    state_list=options.get('state_list',['ACTIVE_EXPERIMENT'])
    state_shuffle=options.get('state_shuffle',[])
    
    state_signals=[]
    for st in state_list:
        print("Adding state {0}".format(st))
        if st=='pupil':
            t=recording.get_signal('pupil')
            if t is None:
                t=resp.epoch_to_signal('XXX')
                t.chans=['pupil']                
            state_signals.append(t)
            
        elif st=='PRE_PASSIVE':
            # only include if post_passive exists
            fpre=(resp.epochs['name']=="PRE_PASSIVE")
            fpost=(resp.epochs['name']=="POST_PASSIVE")
            INCLUDE_PRE_POST=(np.sum(fpre)>0) & (np.sum(fpost)>0)
            if INCLUDE_PRE_POST:
                state_signals.append(resp.epoch_to_signal('PRE_PASSIVE'))
            else:
                t=resp.epoch_to_signal('XXX')
                t.chans=['PRE_PASSIVE']
                state_signals.append(t)
        else:
            state_signals.append(resp.epoch_to_signal(st))
            
    #create state signal with all variables
    state=resp.concatenate_channels(state_signals)
    state.name='state'
    
    # generate average PSTH response to each distinct stimulus
    ff=resp.epochs['name'].str.contains('STIM_')
    stim_names=list(resp.epochs.loc[ff,'name'].unique())
    stim_names.sort()
    
    # extract response to each stim
    r_dict=resp.extract_epochs(stim_names)
    
    # compute mean
    for k,x in r_dict.items():
        r_dict[k]=np.nanmean(x,axis=0)
        
    # replace each response with mean
    r2=resp.replace_epochs(r_dict)
    
    r=resp.select_epoch('REFERENCE').as_continuous().T
    p0=r2.select_epoch('REFERENCE').as_continuous().T
    s=state.select_epoch('REFERENCE').as_continuous().T
    
    ff=np.isfinite(p0)
    r=r[ff,np.newaxis]
    p0=p0[ff,np.newaxis]
    s=s[ff[:,0],:]
    
    for i in state_shuffle:
        print("shuffling {0}".format(state_list[i]))
        s[:,i]=s[np.random.permutation(s.shape[0]),i]
        
    cols=state.chans
    stds=np.std(s,axis=0)
    s=s[:,stds>0] * 1.0
    keepstates,=np.where(stds>0)
    cols=[cols[i] for i in keepstates]
    
    # OLD: normalize s to have mean zero, variance 1
    s=s-np.mean(s,axis=0,keepdims=True)
    s=s/np.std(s,axis=0,keepdims=True)
    s=(s-np.min(s,axis=0,keepdims=True))/2
    # Instead, normalize s to have min zero, max 1
    #s=s-np.min(s,axis=0,keepdims=True)
    #s=s/np.max(s,axis=0,keepdims=True)
    print(np.max(s,axis=0))
    
    spont=r2.select_epoch('PreStimSilence')
    sp=spont.as_continuous().T
    sp=sp[np.isfinite(sp)]
    
    # subtract spont from prediction
    m0=np.mean(sp)
    p0=p0-m0
    
    X=np.concatenate([p0,m0*s,p0*s],axis=1)

    Xlabels=['p0'] + [x.replace("mask: ","").replace(" ","_")+'_bs' for x in cols]+ \
        [x.replace("mask: ","").replace(" ","_")+'_gn' for x in cols]
        
    param_set,pred,xc,xc_set=jack_model(X,r,Xlabels,10)
            
    res={}
    res['cellid']=cellid
    res['batch']=batch
    
    labels=['b0','p0'] + [x.replace("mask: ","").replace(" ","_")+'_bs' for x in state.chans]+ \
            [x.replace("mask: ","").replace(" ","_")+'_gn' for x in state.chans]
    k=np.concatenate((np.arange(0,2),keepstates+2,keepstates+len(state.chans)+2))

    res['labels']=labels
    res['params']=np.zeros([len(labels)])*np.nan
    res['pvalues']=np.zeros([len(labels)])*np.nan
    
    res['params'][k]=np.mean(param_set,axis=1)
    res['pvalues'][k]=1 # results.pvalues
    res['r']=xc
    res['r_jack']=xc_set
    xc_err=np.std(xc_set)/np.sqrt(xc_set.shape[0])
    if SHOW_PLOT:
        if options['plot_ax'] is None:
            plt.figure()
        else:
            plt.axes(options['plot_ax'])
            
        r1=scipy.signal.decimate(r,q=5,axis=0)
        p1=scipy.signal.decimate(pred,q=5,axis=0)
        t=np.arange(0,r1.shape[0])/options["rasterfs"]*5
        plt.plot(t,r1, linewidth=1)
        #plt.plot(p0+m0, linewidth=1)
        plt.plot(t,p1, linewidth=1,alpha=0.5)
        c=cols
        for i in range(0,s.shape[1]):
            x=s[:,i]
            x=x-x.min()
            x=x/x.max()
            
            x=x[2::5]
            
            t=np.arange(0,x.shape[0])/options["rasterfs"]*5
            plt.plot(t,x-(i+1)*1.2)
            
            if res['pvalues'][i+2]<0.01:
                sb='**'
            elif res['pvalues'][i+2]<0.05:
                sb='*'
            else:
                sb=''
            
            if res['pvalues'][i+len(state.chans)+2]<0.01:
                sg='**'
            elif res['pvalues'][i+len(state.chans)+2]<0.05:
                sg='*'
            else:
                sg=''
            plt.text(0,-(i+1)*1.2,"{0} (b {1:.2f}{2} g {3:.2f}{4})".format(
                    c[i],res['params'][keepstates[i]+2],sb,res['params'][keepstates[i]+len(state.chans)+2],sg))
    
        plt.title("Cell {0}/{1} (pred xc={2:.3f}+/-{3:.3f})".format(
                cellid,batch,xc,xc_err))
    
    return res
    

def jack_model(X,r,Xlabels,jack_count=10):
    
    T=X.shape[0]
    
    param_set=np.array([[]])
    pred=np.zeros(r.shape)
    xc_set=np.zeros(jack_count)
    for i in range(0,jack_count):
        ff=np.arange(0,T,jack_count)+i
        ff=ff[ff<T]
        tX=X.copy()
        vX=X[ff,:]
        tX[ff,:]=np.nan
        tr=r[np.isfinite(tX[:,0])]
        tX=tX[np.isfinite(tX[:,0]),:]
        d=pd.DataFrame(data=tX, columns=Xlabels)
        d['r']=tr
        
        formula='r ~ ' + " + ".join(Xlabels)
        
        results = smf.ols(formula, data=d).fit_regularized(alpha=0.002, L1_wt=0.99)
        #results = smf.ols(formula, data=d).fit()
        #results = smf.gls(formula, data=d).fit()
        params=results.params[:,np.newaxis]
        if i==0:
            param_set=params
        else:
            param_set=np.concatenate((param_set,params),axis=1)
        pred[ff,:]=np.matmul(vX,params[1:,:])+params[0,0]
        xc_set[i]=np.corrcoef(pred[ff,:].T,r[ff,:].T)[0,1]
    xc=np.corrcoef(pred.T,r.T)[0,1]
    
    return param_set,pred,xc,xc_set


def scatter_comp(beta,compare_set,xlabels,axs=None):
    if axs is None:
        axs=plt.gca()
        
    mmin=-0.25
    mmax=0.75
        
    i0=[i for i, x in enumerate(xlabels) if x==compare_set[0]]
    i1=[i for i, x in enumerate(xlabels) if x==compare_set[1]]
    plt.plot(np.array([mmin,mmax]),np.array([mmin,mmax]),'k--')
    x=beta[i0,:].T
    y=beta[i1,:].T
    ff=np.isfinite(x) & np.isfinite(y) & (x!=y)
    x=x[ff]
    y=y[ff]
    mx=np.mean(x)
    my=np.mean(y)
    x[x<mmin]=mmin
    x[x>mmax]=mmax
    y[y<mmin]=mmin
    y[y>mmax]=mmax
    print(np.nanstd(x))
    print(np.nanstd(y))
    plt.scatter(x,y)
    plt.plot(mx,my,'rs')
    axs.set_title("mean {0:.3f} v. {1:.3f}".format(mx,my))
    axs.set_aspect('equal','box')
    axs.set_xlabel(compare_set[0])
    axs.set_ylabel(compare_set[1])
        
    
# batches must be defined:
# 271 - A1 NAT all
# 289 - A1 NAT+pupil
# 301- A1 PTD + pupil
# 304- IC TIN + pupil
# 305- IC TIN  
    
if 'batch' not in locals():
    batch=271
    
print("Analyzing batch {0}".format(batch))

options={'rasterfs': 10, 'includeprestim': True, 'stimfmt': 'parm', 
         'chancount': 0, 'pupil': True, 'stim': False,
         'pupil_deblink': True, 'pupil_median': 1,
         'plot_results': True, 'plot_ax': None}


# lowtech: for each state_list, index of entries to shuffle. here,
# it's only coded to test for active vs. passive effects. ie, 
# state_lists[0] is active / passive shuffled
# state_lists[1] is active / passive actual
if batch==289 or batch==294:
    # A1 NAT+pupil, VOC+pupil
    state_lists=[['pupil'],
                 ['pupil']]
    state_shuffles=[[0],[]]
    options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 
             'chancount': 18, 'pupil': True, 'stim': True,
             'pupil_deblink': True, 'pupil_median': 1,
             'plot_results': True, 'plot_ax': None}
#             'runclass': 'VOC',
    
elif batch in [271,272,291]:
    # NAT
    state_lists=[]
    state_shuffles=[[]]
    options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'ozgf', 
             'chancount': 18, 'pupil': False, 'stim': True,
             'plot_results': True, 'plot_ax': None}
elif batch in [259]:
    #  SPN
    state_lists=[]
    state_shuffles=[[]]
    options={'rasterfs': 100, 'includeprestim': True, 'stimfmt': 'envelope', 
             'chancount': 0, 'pupil': False, 'stim': True,
             'plot_results': True, 'plot_ax': None}
    
elif batch==301:
    state_lists=[['ACTIVE_EXPERIMENT'],
                 ['ACTIVE_EXPERIMENT'],
                 ['ACTIVE_EXPERIMENT','pupil'],
                 ['ACTIVE_EXPERIMENT','pupil','MISS_TRIAL','FA_TRIAL']]
    state_shuffles=[[0],[],[],[]]
    
elif batch==304:
    state_lists=[['ACTIVE_EXPERIMENT'],
                 ['ACTIVE_EXPERIMENT'],
                 ['PURETONE_BEHAVIOR','EASY_BEHAVIOR','HARD_BEHAVIOR'],
                 ['PURETONE_BEHAVIOR','EASY_BEHAVIOR','HARD_BEHAVIOR','PRE_PASSIVE'],
                 ['PURETONE_BEHAVIOR','EASY_BEHAVIOR','HARD_BEHAVIOR','PRE_PASSIVE','pupil'],
                 ['PURETONE_BEHAVIOR','EASY_BEHAVIOR','HARD_BEHAVIOR','PRE_PASSIVE','pupil','MISS_TRIAL','FA_TRIAL']]
    state_shuffles=[[0],[],[],[],[],[]]
    
elif batch==305:
    state_lists=[['ACTIVE_EXPERIMENT'],
                 ['ACTIVE_EXPERIMENT'],
                 ['PURETONE_BEHAVIOR','EASY_BEHAVIOR','HARD_BEHAVIOR'],
                 ['PURETONE_BEHAVIOR','EASY_BEHAVIOR','HARD_BEHAVIOR','PRE_PASSIVE'],
                 ['PURETONE_BEHAVIOR','EASY_BEHAVIOR','HARD_BEHAVIOR','PRE_PASSIVE','MISS_TRIAL','FA_TRIAL']]
    state_shuffles=[[0],[],[],[],[]]
    
else:
    raise BaseException("value of batch not valid")

if options['stim']:
    stimfmt=options['stimfmt']
else:
    stimfmt='none'
    

# regenerate or reload recordings as needed
REGEN=False
RELOAD=True
REFIT=False
if REGEN:
    # load data from baphy files and save to nems format
    plt.close('all')
    cell_data=nd.get_batch_cells(batch=batch)
    cellids=list(cell_data['cellid'].unique())
    
    recordings=[]
    save_path="/auto/data/tmp/batch{0}_fs{1}_{2}{3}/".format(batch,options["rasterfs"],stimfmt,options["chancount"])
    for cellid in cellids:
        try:
            recordings=recordings+[nems.utilities.baphy.baphy_load_recording(cellid,batch,options.copy())]
            recordings[-1].save(save_path)
        except:
            print("failed on cell {0}".format(cellid))
elif RELOAD:
    # load data from baphy format
    cell_data=nd.get_batch_cells(batch=batch)
    cellids=list(cell_data['cellid'].unique())
    recordings=[]
    for cellid in cellids:
        save_path="/auto/data/tmp/batch{0}_fs{1}_{2}{3}/{4}".format(batch,options["rasterfs"],stimfmt,options["chancount"],cellid)
        print("Loading from {0}".format(save_path))
        if os.path.exists(save_path):
            trec=nems.recording.Recording.load(save_path)
        else:
            trec=nems.utilities.baphy.baphy_load_recording(cellid,batch,options.copy())
            t_save_path="/auto/data/tmp/batch{0}_fs{1}_{2}{3}/".format(batch,options["rasterfs"],stimfmt,options["chancount"])
            trec.save(t_save_path)
            
        recordings.append(trec)
else:
    # don't load or regen, recordings are already in memory
    print("Assuming data are loaded")



example_list=['TAR010c-06-1','TAR010c-22-1',
          'TAR010c-60-1','TAR010c-44-1',
          'bbl074g-a1','bbl081d-a1','BRT006d-a1','BRT007c-a1',
          'BRT009a-a1','BRT015c-a1','BRT016f-a1','BRT017g-a1']
         
plt.close('all')
cellid=example_list[11]
for i, x in enumerate(cellids):
    cidx=-1
    if x==cellid:
        cidx=i
        
        for j,state_list in enumerate(state_lists):
            options["state_list"]=state_list
            options["state_shuffle"]=state_shuffles[j]
            res=state_gain_model(recordings[cidx],options)
    
    
    
"""

if REFIT:
    plt.close('all')
    res=[]
    res2=[]
    
    for i,cellid in enumerate(cellids):
        print("{0} fitting for cell {1}".format(i,cellid))
        
        res=res+[ptd_gain_model(recordings[i],options)]
        if (batch==304) | (batch==301):
            opt2=options.copy()
            opt2['zero_pupil']=True
            res2=res2+[ptd_gain_model(recordings[i],opt2)]
        else:
            opt2=options.copy()
            opt2['zero_pp']=True
            res2=res2+[ptd_gain_model(recordings[i],opt2)]
else:
    print("Skipping big fit")
#plt.close('all')
beta=np.concatenate([x['params'][2:,np.newaxis] for x in res], axis=1)
sig=np.concatenate([x['pvalues'][2:,np.newaxis] for x in res], axis=1)
beta[np.isfinite(beta) & (beta>1)]=1
beta[np.isfinite(beta) & (beta<-1)]=-1
tbeta=beta.copy()
tbeta[sig>=0.001]=np.nan
tbeta2=beta.copy()
tbeta2[sig<0.001]=np.nan
plt.figure()
plt.subplot(1,2,1)
plt.plot(tbeta,'o')
plt.plot(tbeta2,'.')
xlabels=res[-1]['labels'][2:]
plt.xticks(range(len(xlabels)),xlabels,rotation='vertical',fontsize=6)
plt.title("batch {0} summary".format(batch))

rvalues=np.concatenate([x['rvalues'][:,np.newaxis] for x in res],axis=1)
plt.subplot(1,2,2)
plt.plot(rvalues[1:,:]-rvalues[0:1,:],'.')

plt.tight_layout()


PLOT_EXAMPLES=True
if PLOT_EXAMPLES:
    plt.close('all')

    options["plot_results"]=True
    options["pupil"]=True
    options['zero_pupil']=False
    opt2=options.copy()
    opt2['zero_pupil']=True
    for cellid in example_list:
        cidx=-1
        for i, x in enumerate(cellids):
            if x==cellid:
                cidx=i
                
        if cidx>0:
            print("{0}".format(cellids[cidx]))
            plt.figure()
            options['plot_ax']=plt.subplot(2,1,1)
            ptd_gain_model(recordings[cidx],options)
            opt2['plot_ax']=plt.subplot(2,1,2)
            ptd_gain_model(recordings[cidx],opt2)
            plt.tight_layout()

if (batch==304) | (batch==301):
    plt.figure()

    beta2=np.concatenate([x['params'][2:,np.newaxis] for x in res2], axis=1)
    sig2=np.concatenate([x['pvalues'][2:,np.newaxis] for x in res2], axis=1)
    tbeta=np.concatenate((beta,beta2[3:5,:]),axis=0)
    tsig=np.concatenate((sig,sig2[3:5,:]),axis=0)
    
    txlabels=xlabels+['PRE_NO_PUP_gn','ACTIVE_NO_PUP_gn']
    
    print("{0} {1} {2}".format(txlabels[4],txlabels[5],txlabels[7]))
    for cc in range(0,tbeta.shape[1]):
        print("{0:2d} {1} {2:8.3f} {3:8.3f} {4:8.3f}".format(cc,cellids[cc],
              tbeta[4,cc],tbeta[5,cc],tbeta[7,cc]))
        
    
    axs = plt.subplot(2,2,1)
    compare_set=['ACTIVE_EXPERIMENT_gn','ACTIVE_NO_PUP_gn']
    scatter_comp(tbeta,compare_set,txlabels,axs)
    
    axs = plt.subplot(2,2,2)
    compare_set=['PRE_PASSIVE_gn','PRE_NO_PUP_gn']
    scatter_comp(tbeta,compare_set,txlabels,axs)
    
    axs = plt.subplot(2,2,3)
    compare_set=['ACTIVE_NO_PUP_gn','pupil_gn']
    scatter_comp(tbeta,compare_set,txlabels,axs)
    
    axs = plt.subplot(2,2,4)
    compare_set=['ACTIVE_EXPERIMENT_gn','pupil_gn']
    scatter_comp(beta,compare_set,xlabels,axs)
    
    plt.tight_layout()
else:
    plt.figure()

    beta2=np.concatenate([x['params'][2:,np.newaxis] for x in res2], axis=1)
    sig2=np.concatenate([x['pvalues'][2:,np.newaxis] for x in res2], axis=1)
    tbeta=np.concatenate((beta,beta2[8:11,:]),axis=0)
    tsig=np.concatenate((sig,sig2[8:11,:]),axis=0)
    
    txlabels=xlabels+['PURE_NO_PRE_gn','EASY_NO_PRE_gn','HARD_NO_PRE_gn']

    axs = plt.subplot(2,2,1)
    compare_set=['EASY_BEHAVIOR_gn','HARD_BEHAVIOR_gn']
    scatter_comp(tbeta,compare_set,txlabels,axs)
    
    axs = plt.subplot(2,2,2)
    compare_set=['PURETONE_BEHAVIOR_gn','EASY_BEHAVIOR_gn']
    scatter_comp(tbeta,compare_set,txlabels,axs)
    
    compare_set=['EASY_NO_PRE_gn','HARD_NO_PRE_gn']
    axs = plt.subplot(2,2,3)
    scatter_comp(tbeta,compare_set,txlabels,axs)
    
    compare_set=['PURE_NO_PRE_gn','EASY_NO_PRE_gn']
    axs = plt.subplot(2,2,4)
    scatter_comp(tbeta,compare_set,txlabels,axs)

    plt.tight_layout()

print("Done with batch {0}".format(batch))

"""
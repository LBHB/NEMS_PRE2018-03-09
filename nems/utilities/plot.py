#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utilities.plot.py:  plotting functions called by modules in nems stack

Created on Sep 7 2017

@author: svd
"""

import scipy.signal as sps
import scipy
import numpy as np
import numpy.ma as npma
import matplotlib.pyplot as plt

# set default figsize for pyplots (so we don't have to change each function)
FIGSIZE=(12,4)

#
# PLOTTING FUNCTIONS
#
#TODO: find some way to get the stimuli to resolve correctly for the pupil model stacks,
#since stack.data[1] will have ~2 stimuli, but it is subsequently reshaped to ~240 stimuli
# ---fixed, see except statement in plot_spectrogram --njs 6 July 2017

def plot_spectrogram(m,idx=None,size=FIGSIZE):
    #Moved from pylab to pyplot module in all do_plot functions, changed plots 
    #to be individual large figures, added other small details -njs June 16, 2017
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    reps=out1['repcount']
    ids=m.parent_stack.plot_stimidx
    r=reps.shape[0]
    lis=[]
    for i in range(0,r):
        lis.extend([i]*reps[i])
    new_id=lis[ids]
    if out1['stim'].ndim==3:
        try:
            plt.imshow(out1['stim'][:,m.parent_stack.plot_stimidx,:], aspect='auto', origin='lower', interpolation='none')
        except:
            plt.imshow(out1['stim'][:,new_id,:], aspect='auto', origin='lower', interpolation='none')
        cbar = plt.colorbar()
        cbar.set_label('amplitude')
        # TODO: colorbar is intensity of spectrogram/response, units not clearly specified yet
        plt.xlabel('Time')
        plt.ylabel('Channel')
    else:
        s=out1['stim'][:,new_id]
        #r=out1['resp'][m.parent_stack.plot_stimidx,:]
        pred, =plt.plot(s,label='Average Model')
        #resp, =plt.plot(r,'r',label='Response')
        plt.legend(handles=[pred])
        # TODO: plot time in seconds
        plt.xlabel('Time Step')
        plt.ylabel('Firing rate (a.u.)')
            
    #plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))

def plot_stim(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=str(idx),figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    #c=out1['repcount'][m.parent_stack.plot_stimidx]
    #h=out1['stim'][m.parent_stack.plot_stimidx].shape
    #scl=int(h[0]/c)
    s2=np.squeeze(out1['stim'][:,m.parent_stack.plot_stimidx,:]).transpose()
    print(s2.shape)
    plt.plot(s2)
    #plt.title(m.name+': stim #'+str(m.parent_stack.plot_stimidx))

def pred_act_scatter(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s=out1[m.output_name][m.parent_stack.plot_stimidx,:]
    r=out1['resp'][m.parent_stack.plot_stimidx,:]
    plt.plot(s,r,'ko')
    plt.xlabel("Predicted ({0})".format(m.output_name))
    plt.ylabel('Actual')
    #plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    #plt.title("{0} (r_est={1:.3f}, r_val={2:.3f})".format(m.name,m.parent_stack.meta['r_est'],m.parent_stack.meta['r_val']))
    axes = plt.gca()
    ymin, ymax = axes.get_ylim()
    xmin, xmax = axes.get_xlim()
    if ymin==ymax:
        ymax=ymin+1
    if xmin==xmax:
        xmax=xmin+1
    #print("{0},{1} {2},{3}".format(xmin,xmax,ymin,ymax))
    plt.text(xmin+(xmax-xmin)/50,ymax-(ymax-ymin)/20,"r_est={0:.3f}\nr_val={1:.3f}".format(m.parent_stack.meta['r_est'][0],m.parent_stack.meta['r_val'][0]),
             verticalalignment='top')
    
    
def io_scatter_smooth(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=idx,figsize=size)
    s=m.unpack_data(m.input_name,use_dout=False)
    r=m.unpack_data(m.output_name,use_dout=True)
    r2=m.unpack_data("resp",use_dout=True)
    s2=np.append(s.transpose(),r.transpose(),0)
    s2=np.append(s2,r2.transpose(),0)
    s2=s2[:,s2[0,:].argsort()]
    bincount=np.min([100,s2.shape[1]])
    T=np.int(np.floor(s2.shape[1]/bincount))
    s2=s2[:,0:(T*bincount)]
    s2=np.reshape(s2,[3,bincount,T])
    s2=np.mean(s2,2)
    s2=np.squeeze(s2)
    
    plt.plot(s2[0,:],s2[1,:],'k-')
    plt.plot(s2[0,:],s2[2,:],'k.')
    plt.xlabel("Input ({0})".format(m.input_name))
    plt.ylabel("Output ({0})".format(m.output_name))
    #plt.title("{0}".format(m.name))

def scatter_smooth(m,idx=None,x_name=None,y_name=None,size=FIGSIZE):
    if idx:
        plt.figure(num=idx,figsize=size)
    if not x_name:
        x_name=m.output_name
    if not y_name:
        y_name="resp"
        
    s=m.unpack_data(x_name,use_dout=True)
    r=m.unpack_data(y_name,use_dout=True)
    s2=np.append(s.transpose(),r.transpose(),0)
    s2=s2[:,s2[0,:].argsort()]
    bincount=np.min([100,s2.shape[1]])
    T=np.int(np.floor(s2.shape[1]/bincount))
    s2=s2[:,0:(T*bincount)]
    s2=np.reshape(s2,[2,bincount,T])
    s2=np.mean(s2,2)
    s2=np.squeeze(s2)
    plt.plot(s2[0,:],s2[1,:],'k.')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    #m.parent_stack.meta['r_val']
    #plt.title("{0} (r_est={1:.3f}, r_val={2:.3f})".format(m.name,m.parent_stack.meta['r_est'],m.parent_stack.meta['r_val']))

def pred_act_scatter_smooth(m,idx=None,size=FIGSIZE):
    scatter_smooth(m,idx=idx,size=size,x_name=m.output_name,y_name="resp")

def state_act_scatter_smooth(m,idx=None,size=FIGSIZE):
    scatter_smooth(m,idx=idx,size=size,x_name=m.state_var,y_name="resp")
    
def pred_act_psth(m,size=FIGSIZE,idx=None):
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s=out1['stim'][m.parent_stack.plot_stimidx,:]
    r=out1['resp'][m.parent_stack.plot_stimidx,:]
    pred, =plt.plot(s,label='Predicted')
    act, =plt.plot(r,'r',label='Actual')
    plt.legend(handles=[pred,act])
    #plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time Step')
    plt.ylabel('Firing rate (unitless)')

def pred_act_psth_smooth(m,size=FIGSIZE,idx=None):
    if idx:
        plt.figure(num=idx,figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    s=out1['stim'][m.parent_stack.plot_stimidx,:]
    r=out1['resp'][m.parent_stack.plot_stimidx,:]
        
    box_pts=20
    box = np.ones(box_pts)/box_pts
    s = np.convolve(s, box, mode='same')
    r = np.convolve(r, box, mode='same')
    
    pred, =plt.plot(s,label='Predicted')
    act, =plt.plot(r,'r',label='Actual')
    plt.legend(handles=[pred,act])
    #plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time Step')
    plt.ylabel('Firing rate (unitless)')

def pred_act_psth_all(m,size=FIGSIZE,idx=None):
    if idx:
        plt.figure(num=idx,figsize=size)
    s=m.unpack_data(m.output_name,use_dout=True)
    r=m.unpack_data("resp",use_dout=True)
    s2=np.append(s.transpose(),r.transpose(),0)
    try:
        p=m.unpack_data("pupil",use_dout=True)
        s2=np.append(s2,p.transpose(),0)
        p_avail=True
    except:
        p_avail=False
    
    bincount=np.min([5000,s2.shape[1]])
    T=np.int(np.floor(s2.shape[1]/bincount))
    s2=np.reshape(s2[:,0:(T*bincount)],[3,T,bincount])
    s2=np.mean(s2,1)
    s2=np.squeeze(s2)
    
    pred, =plt.plot(s2[0,:],label='Predicted')
    act, =plt.plot(s2[0,:],'r',label='Actual')
    if p_avail:
        pup, =plt.plot(s2[0,:],'g',label='Pupil')
        plt.legend(handles=[pred,act,pup])
    else:
        plt.legend(handles=[pred,act])
        
    #plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time Step')
    plt.ylabel('Firing rate (unitless)')

def pre_post_psth(m,size=FIGSIZE,idx=None):
    if idx:
        plt.figure(num=idx,figsize=size)
    in1=m.d_in[m.parent_stack.plot_dataidx][m.input_name]
    out1=m.d_out[m.parent_stack.plot_dataidx][m.output_name]
    if len(in1.shape)>2:
        s1=in1[0,m.parent_stack.plot_stimidx,:]
    else:
        s1=in1[m.parent_stack.plot_stimidx,:]
    if len(out1.shape)>2:
        s2=out1[0,m.parent_stack.plot_stimidx,:]
    else:
        s2=out1[m.parent_stack.plot_stimidx,:]
        
    pre, =plt.plot(s1,label='Pre-nonlinearity')
    post, =plt.plot(s2,'r',label='Post-nonlinearity')
    plt.legend(handles=[pre,post])
    #plt.title("{0} (data={1}, stim={2})".format(m.name,m.parent_stack.plot_dataidx,m.parent_stack.plot_stimidx))
    plt.xlabel('Time Step')
    plt.ylabel('Firing rate (unitless)')

def plot_stim_psth(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=str(idx),figsize=size)
    out1=m.d_out[m.parent_stack.plot_dataidx]
    #c=out1['repcount'][m.parent_stack.plot_stimidx]
    #h=out1['stim'][m.parent_stack.plot_stimidx].shape
    #scl=int(h[0]/c)
    s2=out1['stim'][m.parent_stack.plot_stimidx,:]
    resp, =plt.plot(s2,'r',label='Post-'+m.name)
    plt.legend(handles=[resp])
    #plt.title(m.name+': stim #'+str(m.parent_stack.plot_stimidx))
        
  
def plot_strf(m,idx=None,size=FIGSIZE):
    if idx:
        plt.figure(num=idx,figsize=size)
    h=m.coefs
    
    # if weight channels exist and dimensionality matches, generate a full STRF
    try:
        wcidx=find_modules(m.parent_stack,"filters.weight_channels")
        if len(wcidx)>0 and m.parent_stack.modules[wcidx[0]].output_name==m.output_name:
            wcidx=wcidx[0]
        elif len(wcidx)>1 and m.parent_stack.modules[wcidx[1]].output_name==m.output_name:
            wcidx=wcidx[1]
        else:
            wcidx=-1
    except:
        wcidx=-1
        
    if m.name=="filters.fir" and wcidx>=0:
        #print(m.name)
        w=m.parent_stack.modules[wcidx].coefs
        if w.shape[0]==h.shape[0]:
            h=np.matmul(w.transpose(), h)
        
    mmax=np.max(np.abs(h.reshape(-1)))
    plt.imshow(h, aspect='auto', origin='lower',cmap=plt.get_cmap('jet'), interpolation='none')
    plt.clim(-mmax,mmax)
    cbar = plt.colorbar()
    #TODO: cbar.set_label('???')
    #plt.title(m.name)S
    plt.xlabel('Channel') #or kHz?
    # Is this correct? I think so...
    plt.ylabel('Latency')
    
def non_plot(m):
    pass

def raster_data(data,pres,dura,posts,fr):
    s=data.shape
    pres=int(pres)
    dura=int(dura)
    posts=int(posts)
    xpre=np.zeros((s[2],pres*s[1]))
    ypre=np.zeros((s[2],pres*s[1]))
    xdur=np.zeros((s[2],dura*s[1]))
    ydur=np.zeros((s[2],dura*s[1]))
    xpost=np.zeros((s[2],posts*s[1]))
    ypost=np.zeros((s[2],posts*s[1]))
    for i in range(0,s[2]):
        spre=0
        sdur=0
        spost=0
        for j in range(0,s[1]):
            ypre[i,spre:(spre+pres)]=(j+1)*np.clip(data[:pres,j,i],0,1)
            xpre[i,spre:(spre+pres)]=np.divide(np.array(list(range(0,pres))),fr)
            ydur[i,sdur:(sdur+dura)]=(j+1)*np.clip(data[pres:(pres+dura),j,i],0,1)
            xdur[i,sdur:(sdur+dura)]=np.divide(np.array(list(range(pres,(pres+dura)))),fr)
            ypost[i,spost:(spost+posts)]=(j+1)*np.clip(data[(pres+dura):(pres+dura+posts),j,i],0,1)
            xpost[i,spost:(spost+posts)]=np.divide(
                    np.array(list(range((pres+dura),(pres+dura+posts)))),fr)
            spre+=pres
            sdur+=dura
            spost+=posts
    ypre[ypre==0]=None
    ydur[ydur==0]=None
    ypost[ypost==0]=None
    return(xpre,ypre,xdur,ydur,xpost,ypost)

def raster_plot(m,idx=None,size=(12,6)):
    """
    This function generates a raster plot of the data for the specified stimuli.
    It shows the spikes that occur during the actual trial in green, and the background
    spikes in grey. 
    """
    resp=m.parent_stack.unresampled['resp']
    pre=m.parent_stack.unresampled['prestim']
    dur=m.parent_stack.unresampled['duration']
    post=m.parent_stack.unresampled['poststim']
    freq=m.parent_stack.unresampled['respFs']
    #print("{}/{}/{} fs {}".format(pre,dur,post,freq))
    total_bins=(pre+dur+post)*freq.astype(int)
    #print(total_bins)
    if resp.shape[0]<total_bins:
        d=total_bins-resp.shape[0]
        s=list(resp.shape)
        s[0]=d
        resp=np.concatenate((resp,np.zeros(s)),axis=0)
    #print(resp.shape)
    reps=m.parent_stack.unresampled['repcount']
    ids=m.parent_stack.plot_stimidx
    r=reps.shape[0]
    prestim=float(pre)*freq
    duration=float(dur)*freq
    poststim=float(post)*freq
    if m.parent_stack.unresampled['pupil'] is not None:
        lis=[]
        for i in range(0,r):
            lis.extend([i]*reps[i])
        stims=lis[ids]
    else:
        stims=ids
    xpre,ypre,xdur,ydur,xpost,ypost=raster_data(resp,prestim,duration,poststim,freq)
    if idx is not None:
        plt.figure(num=str(stims)+str(100),figsize=size)
    plt.scatter(xpre[stims],ypre[stims],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xdur[stims],ydur[stims],color='g',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xpost[stims],ypost[stims],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.ylabel('Trial')
    plt.xlabel('Time')
    plt.title('Stimulus #'+str(stims))

def sorted_raster(m,idx=None,size=FIGSIZE):
    """
    Creates a raster plot sorted by mean pupil diameter of a given trial
    """
    resp=m.parent_stack.unresampled['resp']
    pre=m.parent_stack.unresampled['prestim']
    dur=m.parent_stack.unresampled['duration']
    post=m.parent_stack.unresampled['poststim']
    freq=m.parent_stack.unresampled['respFs']
    reps=m.parent_stack.unresampled['repcount']
    r=reps.shape[0]
    prestim=float(pre)*freq
    duration=float(dur)*freq
    poststim=float(post)*freq
    pup=m.parent_stack.unresampled['pupil']
    idi=m.parent_stack.plot_stimidx
    lis=[]
    for i in range(0,r):
            lis.extend([i]*reps[i])
    ids=lis[idi]
    b=np.nanmean(pup[:,:,ids],axis=0)
    b=np.nan_to_num(b)
    bc=np.asarray(sorted(zip(b,range(0,len(b)))),dtype=int)
    bc=bc[:,1]
    resp[:,:,ids]=resp[:,bc,ids]
    xpre,ypre,xdur,ydur,xpost,ypost=raster_data(resp,prestim,duration,poststim,freq)
    if idx is not None:
        plt.figure(num=str(ids)+str(100),figsize=size)
    plt.scatter(xpre[ids],ypre[ids],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xdur[ids],ydur[ids],color='g',s=(0.5*np.pi)*2,alpha=0.6)
    plt.scatter(xpost[ids],ypost[ids],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
    plt.ylabel('Trial')
    plt.xlabel('Time')
    plt.title('Sorted by Pupil: Stimulus #'+str(ids))
    

#
# Other support functions
#

def shrinkage(mH,eH,sigrat=1,thresh=0):

    smd=np.abs(mH)/(eH+np.finfo(float).eps*(eH==0)) / sigrat

    if thresh:
       hf=mH*(smd>1)
    else:
       smd=1-np.power(smd,-2)
       smd=smd*(smd>0)
       #smd[np.isnan(smd)]=0
       hf=mH*smd
    
    return hf

def concatenate_helper(stack,start=1,**kwargs):
    """
    Helper function to concatenate the nest list in the validation data. Simply
    takes the lists in stack.data if ['est'] is False and concatenates all the 
    subarrays.
    """
    try:
        end=kwargs['end']
    except:
        end=len(stack.data)
    for k in range(start,end):
        #print('start loop 1')
        #print(len(stack.data[k]))
        for n in range(0,len(stack.data[k])):
            #print('start loop 2')
            if stack.data[k][n]['est'] is False:
                #print('concatenating')
                if stack.data[k][n]['stim'][0].ndim==3:
                    stack.data[k][n]['stim']=np.concatenate(stack.data[k][n]['stim'],axis=1)
                else:
                    stack.data[k][n]['stim']=np.concatenate(stack.data[k][n]['stim'],axis=0)
                stack.data[k][n]['resp']=np.concatenate(stack.data[k][n]['resp'],axis=0)
                try:
                    stack.data[k][n]['pupil']=np.concatenate(stack.data[k][n]['pupil'],axis=0)
                except:
                    stack.data[k][n]['pupil']=None
                try:
                    stack.data[k][n]['replist']=np.concatenate(stack.data[k][n]['replist'],axis=0)
                except:
                    stack.data[k][n]['replist']=[]
                try:
                    stack.data[k][n]['repcount']=np.concatenate(stack.data[k][n]['repcount'],axis=0)
                except:
                    pass
                if 'stim2' in stack.data[k][n]:
                    if stack.data[k][n]['stim2'][0].ndim==3:
                        stack.data[k][n]['stim2']=np.concatenate(stack.data[k][n]['stim2'],axis=1)
                    else:
                        stack.data[k][n]['stim2']=np.concatenate(stack.data[k][n]['stim2'],axis=0)
            else:
                pass
            
def thresh_resamp(data,resamp_factor,thresh=0,ax=0):
    """
    Helper function to apply an FIR downsample to data. If thresh is specified, 
    the function will send all values in data below thresh to 0; this is often 
    useful to reduce the ringing caused by FIR downsampling.
    """
    resamp=sps.decimate(data,resamp_factor,ftype='fir',axis=ax,zero_phase=True)
    s_indices=resamp<thresh
    resamp[s_indices]=0
    return resamp

def stretch_trials(data):
    """
    Helper function to "stretch" trials to be treated individually as stimuli. 
    This function is used when it is not desirable to average over the trials
    of the stimuli in a dataset, such as when the effects of state variables such 
    as pupil diameter are being explored.
    
    'data' should be the imported data dictionary, and must contain 'resp',
    'stim', 'pupil', and 'repcount'. Note that 'stim' should be formatted as 
    (channels,stimuli,time), while 'resp' and 'pupil' should be formatted as
    (time,trials,stimuli). These are the configurations used in the default 
    loading module nems.modules.load_mat
    """
    #r=data['repcount']
    s=data['resp'].shape # time X rep X stim
    
    # stack each rep on top of each other
    resp=np.transpose(data['resp'],(0,2,1)) # time X stim X rep
    resp=np.transpose(np.reshape(resp,(s[0],s[1]*s[2]),order='F'),(1,0))
    
    #data['resp']=np.transpose(np.reshape(data['resp'],(s[0],s[1]*s[2]),order='C'),(1,0)) #Interleave
    #mask=np.logical_not(npma.getmask(npma.masked_invalid(resp)))
    #R=resp[mask]
    #resp=np.reshape(R,(-1,s[0]),order='C')
    try:
        # stack each rep on top of each other -- identical to resp
        pupil=np.transpose(data['pupil'],(0,2,1))
        pupil=np.transpose(np.reshape(pupil,(s[0],s[1]*s[2]),order='F'),(1,0))
        #P=pupil[mask]
        #pupil=np.reshape(P,(-1,s[0]),order='C')
        #data['pupil']=np.transpose(np.reshape(data['pupil'],(s[0],s[1]*s[2]),order='C'),(1,0)) #Interleave
    except ValueError:
        pupil=None
        
    # copy stimulus as many times as there are repeats -- same stacking as resp??
    stim=data['stim']
    for i in range(1,s[1]):
        stim=np.concatenate((stim,data['stim']),axis=1)
    
    # construct list of which stimulus idx was played on each trial
    # should be able to do this much more simply!
    lis=np.mat(np.arange(s[2])).transpose()
    replist=np.repeat(lis,s[1],axis=1)
    replist=np.reshape(replist.transpose(),(-1,1))
    
#    Y=data['stim'][:,0,:]
#    stim=np.repeat(Y[:,np.newaxis,:],r[0],axis=1)
#    for i in range(1,s[2]):
#        Y=data['stim'][:,i,:]
#        Y=np.repeat(Y[:,np.newaxis,:],r[i],axis=1)
#        stim=np.append(stim,Y,axis=1)
#    lis=[]
#    for i in range(0,r.shape[0]):
#        lis.extend([i]*data['repcount'][i])
#    replist=np.array(lis)
    return stim, resp, pupil, replist


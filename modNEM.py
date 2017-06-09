#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 10:48:40 2017

@author: shofer
"""
import numpy as np
import copy
import math as mt
import scipy as sp
import scipy.signal as sps
import scipy.io as si
import matplotlib as mp
import matplotlib.pyplot as plt
import importlib as il



class FERReT:

    #This initializes the FERReT object, as well as loads the data from the .mat datafile. 
    #Data can also be entered manually, if desired, but metadata must then be entered as well
    def __init__(self,data=None,metadata=None,batch=None,cellid=None,n_coeffs=20,base=0,queue=('input_log','FIR','pupil_gain'),
                 fit_param={'input_log':['log'],'FIR':['coeffs','base'],'ON':['nonlin'],'pupil_gain':['pupil']},
                            thresh=0.5):
        self.batch=batch
        self.cellid=cellid
        self.queue=queue
        ##LOAD DATA FROM .mat FILE
        if data==None:
            self.data=dict.fromkeys(['stim','resp','pup'])
            self.meta=dict.fromkeys(['stimf','respf','iso','prestim','duration','poststim'])
            datapath='/auto/users/shofer/data/'
            file=datapath+'/batch'+str(self.batch)+'/'+self.cellid#+'.mat'
            matdata = si.loadmat(file,chars_as_strings=True)
            m=matdata['data'][0][0]
            self.data['resp']=m['resp_raster']
            self.data['stim']=m['stim']
            self.data['pup']=m['pupil']
            self.meta['stimf']=m['stimfs'][0][0]
            self.meta['respf']=m['respfs'][0][0]
            self.meta['iso']=m['isolation'][0][0]
            self.meta['prestim']=m['tags'][0]['PreStimSilence'][0][0][0]
            self.meta['poststim']=m['tags'][0]['PostStimSilence'][0][0][0]
            self.meta['duration']=m['tags'][0]['Duration'][0][0][0]
        else:
            self.data=data
            self.meta=metadata
        self.ins=copy.deepcopy(self.data)  #Untouched copy of input data, useful raster plots
        self.preserve=dict.fromkeys(['stim','resp','pup']) #filled with resampled data
        self.train=dict.fromkeys(['stim','resp','pup'])
        self.val=dict.fromkeys(['stim','resp','pup'])
        #Set up coefficients. Will almost definitely change
        self.dims=self.data['stim'].shape[0]
        self.shapes=self.data['resp'].shape #(T,R,S)
        self.n_coeffs=n_coeffs
        self.base=np.zeros([1,1])
        self.fit_param=fit_param
        self.mse=float(1.0)
        self.log=np.ones([1,1])
        self.coeffs=np.zeros([self.dims,self.n_coeffs])
        self.nonlin=np.ones([1,4]) #init conditions for DEXP nonlin, maybe change?
        self.nonlin[0][1]=0 #init conditions for DEXP nonlin, maybe change?
        self.nonlin[0][3]=0 #init conditions for DEXP nonlin, maybe change?
        self.pred=np.zeros(self.data['resp'].shape)
        self.current=np.zeros(self.data['resp'].shape)
        self.pupil=np.zeros([1,4])
        self.pupil[0][1]=1
        self.nopupil=np.array([[0,1]])
        self.thresh=thresh
        for i in range(0,self.data['pup'].shape[2]):
            arrmin=np.nanmin(self.data['pup'])
            arrmax=np.nanmax(self.data['pup'])
            if arrmin<0:
                arrmin=0
        self.cutoff=thresh*(arrmax+arrmin)
        self.pupgain_coeffs=[1,1,0,0]
        #Importation module
        self.impdict=dict.fromkeys(queue)
        for j in queue:
           self.impdict[j]=il.import_module('testpack.'+j)
        

        
    #Testing dynamic imports
    def testtest(self):
        getattr(self.impdict['import_test'],'fun_fn')(self)
        getattr(self.impdict['astoria'],'astoria')(self)


##DATA SETUP##
###############################################################################
      
    #Resamples data to new frequency. Should be called before reshape_repetitions
    def data_resample(self,newHz=50,noise_thresh=0.04):
        sHz=self.meta['stimf']
        rHz=self.meta['respf']          
        for i,j in self.data.items():
            if i=='stim':
                resamp_fact=int(sHz/newHz)
                resamp=sps.decimate(j,resamp_fact,ftype='fir',axis=1,zero_phase=True)
                s_indices=resamp<noise_thresh
                resamp[s_indices]=0
                self.data[i]=resamp
                self.preserve[i]=resamp
            else:
                resamp_fact=int(rHz/newHz)
                resamp=sps.decimate(j,resamp_fact,ftype='fir',axis=0,zero_phase=True)
                s_indices=resamp<noise_thresh
                resamp[s_indices]=0
                self.data[i]=resamp
                self.preserve[i]=resamp
        self.shapes=copy.deepcopy(self.data['resp'].shape)
        print("Slow it down! Resampling stimulus from "+str(sHz)+"Hz and response from "
              +str(rHz)+"Hz to "+str(newHz)+"Hz.")
        
    #Reshapes response and, if called in pupil subclass, pupil arrays to T*R x S matrix.
    #CALL AFTER CALLING DATA RESAMPLE
    #If the data is to be resampled, should call data_resample first
    def reshape_repetitions(self):
        s=copy.deepcopy(self.data['resp'].shape)
        for i,j in self.data.items():
            if i in ('resp','pup'):
                self.data[i]=copy.deepcopy(np.reshape(j,(s[0]*s[1],s[2]),order='F'))
            #if i=='stim':
                #self.data[i]=np.tile(j,(1,s[1],1))
        print("Reshaping TxRxS arrays to T*RxS arrays")

                
                
    #SHOULD  BE CALLED AFTER RESHAPE REPETITIONS
    #This function creates training and validation datasets simply by taking the 
    #fraction "valsize" of stimuli of the end of the data. Could randomize the section 
    #the validation set is taken from relatively easily.
    #To be replaced later with a jackknifing routine
    def create_datasets(self,valsize=0.05):
        trainlist=[]
        vallist=[]
        for k,j in self.data.items():
            if k=='stim':
                s=j.shape
                spl=mt.ceil(s[2]*(1-valsize))
                trainlist=j[:,:,:spl]
                vallist=j[:,:,spl:]
                self.train[k]=copy.deepcopy(trainlist)
                self.val[k]=copy.deepcopy(vallist)
            else:
                s=j.shape
                spl=mt.ceil(s[1]*(1-valsize))
                trainlist=j[:,:spl]
                vallist=j[:,spl:]
                self.train[k]=copy.deepcopy(trainlist)
                self.val[k]=copy.deepcopy(vallist)
        print("Creating training and validation datasets.") 
        
    """ THESE FUNCTIONS ARE NOW IN SEPARATE MODULES. External modules should be
    saved as function_name.py, where function_name is the name of the function.
##FUNCTIONS##
############################################################################### 
    def input_log(obj,data):
        #X=copy.deepcopy(self.train['stim'])
        X=data
        v1=obj.log[0,0]
        output=np.log(X+v1)
        obj.train['stim']=output
        return(output)
    
   ##Modeling Functions
   ############################################################################
    def FIR(obj,data): 
        #X=copy.deepcopy(self.train['stim'])
        X=copy.deepcopy(data)
        s=X.shape
        X=np.reshape(X,[s[0],-1])
        for i in range(0,s[0]):
            y=np.convolve(X[i,:],obj.coeffs[i,:])
            X[i,:]=y[0:X.shape[1]]
        X=X.sum(0)+obj.base
        obj.current=np.reshape(X,s[1:])
        obj.pred=copy.deepcopy(obj.current)
        return(obj.current)
    
    #def parametrized
    
    ###########################################################################
    
    def pupil_gain(obj,**kwargs):
        #ins=copy.deepcopy(self.pred)
        ins=kwargs['data'] #data should be self.pred
        #pups=copy.deepcopy(self.train['pup'])
        pups=kwargs['pupdata']
        d0=obj.pupil[0,0]
        g0=obj.pupil[0,1]
        d=obj.pupil[0,2]
        g=obj.pupil[0,3]
        output=d0+(d*pups)+(g0*ins)+g*np.multiply(pups,ins)
        obj.current=output
        return(output)
    """
    #This function applies an offset and a scalar gain to the input "prediction"
    #This is for use in comparing pupil effect, and will most likely not be used
    #in the final model fitting
    def pupil_no_gain(obj,**kwargs):
        #ins=copy.deepcopy(self.pred)
        ins=kwargs['data'] #data should be self.pred
        d0=obj.nopupil[0,0]
        g0=obj.nopupil[0,1]
        output=d0+(g0*ins)
        obj.current=output
        return(output)
    
 
    #Both of these nonlinearities are giving me issues with fitting. They both 
    #tend to send the entire function to a single value, even with a constrained fit
    def ON(obj,**kwargs):  #DEXP output nonlinearity. Working on best way to optimize.
        #ins=copy.deepcopy(self.pred)
        ins=kwargs['data'] #data should be self.pred
        v1=obj.nonlin[0,0]
        v2=obj.nonlin[0,1]
        v3=obj.nonlin[0,2]
        v4=obj.nonlin[0,3]
        output=v1-v2*np.exp(-np.exp(v3*(ins-v4)))
        obj.current=output
        return(output)
    
    def tanhON(obj,**kwargs):
        #ins=copy.deepcopy(self.pred)
        ins=kwargs['data'] #data should be self.pred
        v1=obj.nonlin[0,0]
        v2=obj.nonlin[0,1]
        v3=obj.nonlin[0,2]
        output=v1*np.tanh(v2*ins-v3)+v1
        obj.current=output
        return(output)
    #Need to implement more general nonlinearity module:
    #define one module and several possible functions or just define several modules?
    
###############################################################################

        
    def err(self,tile=False): #calculates Mean Square Error
        E=0
        P=0
        mse=0
        if tile==True:
            reps=self.shapes[1]
            tiled=np.tile(self.current,(1,reps,1))
            E+=np.sum(np.square(tiled-self.train['resp']))
            P=np.sum(np.square(self.train['resp']))
        else:
            E+=np.sum(np.square(self.current-self.train['resp']))
            P=np.sum(np.square(self.train['resp']))
        mse=E/P
        self.mse=mse
        return(mse)
    
    #Converts fit parameters to a single vector, to be used with a "cost function"
    def fit_to_phi(self,to_fit): #to_fit should be formatted ['par1','par2',] etc
        phi=[]
        for k in to_fit:
            g=getattr(self,k).flatten()
            phi=np.append(phi,g)
        return(phi)
    
    #Converts single fit vector back to fit parameters so model can be calculated
    def phi_to_fit(self,phi,to_fit): #to_fit should be formatted ['par1','par2',] etc
        st=0
        for k in to_fit:
            s=getattr(self,k).shape
            setattr(self,k,phi[st:(st+np.prod(s))].reshape(s))
            st+=np.prod(s)
     
    #Normalizes data on a 0 to 1 range, just normalizing over time and trial for now.
    #May change in the future. HAS NOT BEEN UPDATED RECENTLY
    #Input data should be formatted as (Time,Repetition,Stimulus), and should be a numpy array
    def normalize_0to1(data):
        s=data.shape
        arrmin=0
        arrmax=0
        scale=1
        output=np.zeros(s)
        for i in range(0,s[2]):
            arrmin=np.nanmin(data[:,:,i])
            arrmax=np.nanmax(data[:,:,i])
            scale=1/(arrmax-arrmin)
            output[:][:][i]=scale*(data-arrmin)
        return(output)
           
            
    #Converts continuous pupil data to a discrete state variable array of 1 & 0 ~~leaves NaNs in place~~
    #Essentially a worse version of pupil_sort and state_PSTH_data taken in combo, but maybe useful?
    def state_variable_set(self):
        state=copy.deepcopy(self.data['pup'])
        s=state.shape
        for i in range(0,s[0]):
            for j in range(0,s[1]):
                for k in range(0,s[2]):
                    if state[i,j,k]<self.cutoff[k]:
                        state[i][j][k]=0
                    elif state[i,j,k]>=self.cutoff[k]:
                        state[i][j][k]=1
        self.state=state
        return(state)
    
    #Apply a Gaussian(?) nonlinearity to continuous pupil data:
    #def Gauss_pupil_gain(self,gain=1):
        #ins=copy.deepcopy(self.pred)
        #mu=self.pupil[0,0]
        #sigma=self.pupil[0,1]
        #offset=self.pupil[0,2]
        #outs=(1/(sigma*mt.sqrt(2*mt.pi)))*np.exp(-(np.square(ins-mu)/(2*(sigma^2))))+offset
    ##NOT DONE, FOCUS ON LINEAR GAIN, ALSO THIS DOESN'T MAKE MUCH SENSE
    
        
   #This section chops the response data into chunks based on pupil diameter using
   # a simple threshold so that the difference in large and small pupil firing rate
   #can be seen
   #This is not necessary for fitting but is good for data visualization
   ############################################################################
   
   #This extracts the trial number for trials where the pupil diameter was greater than some
   #fraction of the maximum pupil diameter.
    def pupil_sort(self,stimnum=0):
        s=self.data['pup'][:,:,stimnum].shape
        cutoff=self.cutoff
        above=[]
        below=[]
        for j in range(0,s[1]):
            if np.nanmin(self.data['pup'][:,j,stimnum])>=cutoff:
                above.append(j)
            else:
                below.append(j)          
        return(above,below)
        
        
        
    def state_PSTH_data(self,above,below,stimnum=0):
        resp=self.data['resp'][:,:,0]
        high=[]
        low=[]
        la=len(above)
        lb=len(below)
        for i in range(0,la):
            j=above[i]
            high.append(resp[:,j])
        for i in range(0,lb):
            j=below[i]
            low.append(resp[:,j])
        high=np.array(high)
        high=np.transpose(high)
        low=np.array(low)
        low=np.transpose(low)
        return(high,low)
        
        
        
##FITTERS##
###############################################################################
    
    def basic_min(self,function,params,routine='L-BFGS-B',maxit=50000,tiled=True):
        def cost_fn(phi):
            self.phi_to_fit(phi,to_fit=params)
            if function=='FIR': #Modeling fns should be fed ['stim']
                getattr(self.impdict[function],function)(self,self.train['stim'])
            else: #Other fns should be fed pred, the output from modeling functions
                getattr(self.impdict[function],function)(self,data=self.pred,pupdata=self.train['pup'])
            mse=self.err(tiled)
            cost_fn.counter+=1
            if cost_fn.counter % 10000==0:
                print('Eval #'+str(cost_fn.counter))
                print('MSE='+str(mse))
            return(mse)
        opt=dict.fromkeys(['maxiter'])
        opt['maxiter']=int(maxit)
        if function=='tanhON':
            cons=({'type':'ineq','fun':lambda x:np.array([x[0]-0.01,x[1]-0.01,-x[2]-1])})
            routine='COBYLA'
        else:
            cons=()
        phi0=self.fit_to_phi(to_fit=params) 
        cost_fn.counter=0
        phiout=sp.optimize.minimize(cost_fn,phi0,method=routine,
                                    constraints=cons,options=opt)
        #self.pred=copy.deepcopy(self.current)
        return(phiout['x'])
    
    #def basinhopping_min
        
    
    
    
    
##PLOTTING## Want to make these function independent of FERReT class
###############################################################################
    #Generates a raster plot of the data for the specified stimuli
    def raster_plot(self,stims='all'):
        ins=self.ins['resp']
        pre=self.meta['prestim']
        dur=self.meta['duration']
        post=self.meta['poststim']
        freq=self.meta['respf']
        prestim=float(pre)*freq
        duration=float(dur)*freq
        poststim=float(post)*freq
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
        xpre,ypre,xdur,ydur,xpost,ypost=raster_data(ins,prestim,duration,poststim,freq)
        ran=[]
        rs=xpre.shape
        if stims=='all':
            ran=range(0,rs[0])
        elif isinstance(stims,int):
            ran=range(stims,stims+1)
        else:
            ran=range(stims[0],stims[1]+1)
        for i in ran:
            plt.figure(i,figsize=(12,6))
            plt.scatter(xpre[i],ypre[i],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
            plt.scatter(xdur[i],ydur[i],color='g',s=(0.5*np.pi)*2,alpha=0.6)
            plt.scatter(xpost[i],ypost[i],color='0.5',s=(0.5*np.pi)*2,alpha=0.6)
            plt.ylabel('Trial')
            plt.xlabel('Time')
            plt.title('Stimulus #'+str(i))
    

    def plot_pred_resp(self,data,stims='all',trials=False):
        preds=data['predicted']
        resps=data['resp']
        sr=resps.shape
        if stims=='all' and trials==False:
            ran=range(0,sr[1])
        elif stims=='all':
            ran=range(0,sr[2])
        elif isinstance(stims,int):
            ran=range(stims,stims+1)
        else:
            ran=range(stims[0],stims[1]+1)
        for i in ran:
            if trials==False:
                plt.figure(i,figsize=(12,4))
                plt.plot(preds[:,i])
                plt.plot(resps[:,i],'g')
                plt.ylabel('Firing Rate')
                plt.xlabel('Time Step')
                plt.title('Stimulus #'+str(i))
            elif isinstance(trials,int):
                plt.figure((str(i)+str(trials)),figsize=(12,4))
                plt.plot(preds[:,trials,i])
                plt.plot(resps[:,trials,i],'g')
                plt.ylabel('Firing Rate')
                plt.xlabel('Time Step')
                plt.title('Stimulus #'+str(i)+', Trial #'+str(trials))
            else:
                for j in range(trials[0],trials[1]+1):
                    plt.figure((str(i)+str(j)),figsize=(12,4))
                    plt.plot(preds[:,j,i])
                    plt.plot(resps[:,j,i],'g')
                    plt.ylabel('Firing Rate')
                    plt.xlabel('Time Step')
                    plt.title('Stimulus #'+str(i)+', Trial #'+str(j))
                    
    def heatmap(self,model='FIR'):
        arr=getattr(self,self.fit_param[model][0])
        plt.figure(figsize=(12,4))
        plt.imshow(arr)
        plt.colorbar()
        
              
        
##FULL MODULES##
###############################################################################
#This module compares the effects of a pupil gain module on a "perfect" model created
    #by averaging over all trials for a stimulus. This module uses pre-existing components,
    #and will effect the properties of the object, though not irreversibly
    def pupil_comparison(self):
        self.data_resample(newHz=50,noise_thresh=0.04)
        resps=copy.deepcopy(self.data['resp'])  #not sure if training or all data should be used here
        s=self.shapes
        avgs=np.nanmean(resps,axis=1)
        avgs=np.tile(avgs,(s[1],1))
        self.reshape_repetitions()
        self.create_datasets(valsize=0.0)
        self.pred=avgs
        self.basic_min(function='pupil_gain',params=['pupil'],routine='BFGS',tiled=False)
        gainmse=copy.deepcopy(self.mse)
        print(self.pupil)
        self.pred=avgs
        self.basic_min(function='pupil_no_gain',params=['nopupil'],routine='BFGS',tiled=False)
        nogainmse=copy.deepcopy(self.mse)
        print(self.nopupil)
        rat=np.divide(nogainmse,gainmse)
        self.data=copy.deepcopy(self.ins) #Replace modified data with original data 
        return(rat)
        
    #The first element in a queue should always be a modeling function, whether
    #it's the FIR filter, factorized filter, parametrized filter, or some user 
    #defined model. The first element in the queue is called first, and translates 
    #the input stimulus data into a "prediction". The subsequent order of the modules
    #does not matter as far as program functionality, but may effect fitting performance
    def run_fit(self,validation=0.05,reps=1):
        self.data_resample(newHz=50,noise_thresh=0.04)
        self.reshape_repetitions()
        self.create_datasets(valsize=validation)
        self.log[0][0]=10
        self.impdict['input_log'].input_log(self,self.train['stim'])
        for i in range(0,reps):
            print('Repetition #'+str(i+1))
            for j in self.queue[1:len(self.queue)]:
                if j=='pupil_gain':
                    R=self.shapes[1]
                    sp=copy.deepcopy(self.pred.shape)
                    self.pred=np.tile(self.pred,(R,1))
                    self.basic_min(function=j,params=self.fit_param[j],tiled=False)
                    self.pred=self.pred[:sp[0],:]
                else:
                    self.basic_min(function=j,params=self.fit_param[j])
                #self.basic_min('FIR',['base','coeffs'],tiled=True)
                #R=self.shapes[1]
                #self.pred=np.tile(self.pred,(1,R,1))
                #self.basic_min('pupil_gain',['pupil'],routine='BFGS')
            print(self.mse)
 
    def assemble(self,queue=('FIR','pupil_gain'),avgresp=False,useval=True,save=False,filepath=None):
        def shape_back(ins,origdim):
            s=ins.shape
            outs=np.reshape(ins,(origdim[0],origdim[1],s[1]),order='F')
            return(outs)
        dats=dict.fromkeys(['stim','resp','pup','predicted'])
        if useval==True:
            for i,j in self.val.items():
                if i in ('resp','pup'):
                    dats[i]=shape_back(j,origdim=self.shapes)
                elif i=='stim':
                    dats[i]=self.input_log(j)
        else:
            for i,j in self.train.items():
                if i in ('resp','pup'):
                    dats[i]=shape_back(j,origdim=self.shapes)
                else:
                    dats[i]=j
        getattr(self,queue[0])(dats['stim'])
        for k in queue[1:len(queue)]:
            if k=='pupil_gain':
                sp=dats['pup'].shape
                tiled=np.transpose(np.tile(self.pred,(1,1,sp[1])),axes=(1,2,0))
                dats['predicted']=self.pupil_gain(data=tiled,pupdata=dats['pup'])
            else:
                dats['predicted']=getattr(self,k)(data=self.pred)
        #getattr(self,k)(dats['stim'])
        #if avgresp==True:
            #dats['resp']=np.nanmean(dats['resp'],axis=1)
            #dats['predicted']=self.pred
        #else:
            #sp=dats['pup'].shape
            #tiled=np.transpose(np.tile(self.pred,(1,1,sp[1])),axes=(1,2,0))
            #dats['predicted']=self.pupil_gain(data=tiled,pupdata=dats['pup'])
        if save==True:
            np.save(filepath,dats)
        return(dats)
        
    
    
    
    
    
    
    
    
        
        
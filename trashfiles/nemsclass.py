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
import importlib as il



class FERReT:

    #This initializes the FERReT object, as well as loads the data from the .mat datafile. 
    #Data can also be entered manually, if desired, but metadata must then be entered as well
    def __init__(self,filepath=None,imp_type=None,data=None,metadata=None,newHz=50,
                 keyword='lognn_fir15_pupgain',thresh=0.5):
        """
        This initializes the FERReT object. If no data is entered manually, this will
        import the data as specified by imp_type. Initiating the object also imports
        the functions specified in the model keyword/queue. 
        Data can also be entered manually, if desired, but metadata must then be entered as well
        """
        self.file=filepath
        queue=keyword.split('_')
        self.queue=queue
        ##LOAD DATA FROM .mat FILE
        if data is None:
            imp=il.import_module('imports_pack.'+imp_type)
            self.data,self.meta=getattr(imp,imp_type+'_import')(self.file)
            print('Data imported successfully')
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
        self.newHz=50
        self.mse=float(1.0)
        self.pred=np.zeros(self.data['resp'].shape)
        self.current=np.zeros(self.data['resp'].shape)
        self.thresh=thresh
        if self.data['pup'] is not None:
            for i in range(0,self.data['pup'].shape[2]):
                arrmin=np.nanmin(self.data['pup'])
                arrmax=np.nanmax(self.data['pup'])
                if arrmin<0:
                    arrmin=0
            self.cutoff=thresh*(arrmax+arrmin)
        #Import functions from queue:
        self.impdict=dict.fromkeys(queue)
        self.fit_param=dict.fromkeys(queue)
        for j in queue:
           self.impdict[j]=il.import_module('function_pack.'+j)
           print(j+' module imported successfully.')
           self.fit_param[j]=getattr(self.impdict[j],'create_'+j)(self)
        


##DATA SETUP##
###############################################################################
      

    def data_resample(self,noise_thresh=0.04):
        """
        Resamples data to new frequency, where the new frequency is given by self.newHz.
        It also implements a simple cutoff on the decimated data, to alleviate ringing
        effects caused by the decimation. Should be called before reshape_repetitions
        """
        sHz=self.meta['stimf']
        rHz=self.meta['respf']
        newHz=self.newHz          
        if self.data['pup'] is not None:
            test=('pup','resp')
        else:
            test=('resp')
        for i,j in self.data.items():
            if i=='stim':
                resamp_fact=int(sHz/newHz)
                resamp=sps.decimate(j,resamp_fact,ftype='fir',axis=1,zero_phase=True)
                s_indices=resamp<noise_thresh
                resamp[s_indices]=0
                self.data[i]=resamp
                self.preserve[i]=resamp
            elif i in test:
                resamp_fact=int(rHz/newHz)
                resamp=sps.decimate(j,resamp_fact,ftype='fir',axis=0,zero_phase=True)
                s_indices=resamp<noise_thresh
                resamp[s_indices]=0
                self.data[i]=resamp
                self.preserve[i]=resamp
        self.shapes=copy.deepcopy(self.data['resp'].shape)
        print("Slow it down! Resampling stimulus from "+str(sHz)+"Hz and response from "
              +str(rHz)+"Hz to "+str(newHz)+"Hz.")
        


    def reshape_repetitions(self):
        """
        Reshapes response, stimulus, and pupil arrays to T*R x S matrix.
        CALL AFTER CALLING DATA RESAMPLE
        Only call if there is pupil data. If not, call resp_avg
        """     
        s=copy.deepcopy(self.data['resp'].shape)
        for i,j in self.data.items():
            if i in ('resp','pup'):
                self.data[i]=copy.deepcopy(np.reshape(j,(s[0]*s[1],s[2]),order='F'))
            if i=='stim':
                self.data[i]=np.tile(j,(1,s[1],1))
        print("Reshaping TxRxS arrays to T*RxS arrays")
    

    def resp_avg(self):
        """
        Averages response to stimuli across all trials of that stimulus. Use when 
        there is not state variable data, as it will allow for fitting with much
        smaller data sets, resulting in a faster (and potentially more accurate)
        fit
        CALL AFTER CALLING DATA RESAMPLE
        """
        self.data['resp']=np.nanmean(self.data['resp'],axis=1)
        


    def create_datasets(self,valsize=0.05):
        """
        SHOULD  BE CALLED AFTER RESHAPE REPETITIONS
        This function creates training and validation datasets simply by taking the 
        fraction "valsize" of stimuli of the end of the data. Could randomize the section 
        the validation set is taken from relatively easily.
        To be replaced later with a jackknifing routine
        """
        
        trainlist=[]
        vallist=[] 
        if self.data['pup'] is not None:
            test=('pup','resp')
        else:
            test=('resp')
        for k,j in self.data.items():
            if k=='stim':
                s=j.shape
                spl=mt.ceil(s[2]*(1-valsize))
                trainlist=j[:,:,:spl]
                vallist=j[:,:,spl:]
                self.train[k]=copy.deepcopy(trainlist)
                self.val[k]=copy.deepcopy(vallist)
            elif k in test:
                s=j.shape
                spl=mt.ceil(s[1]*(1-valsize))
                trainlist=j[:,:spl]
                vallist=j[:,spl:]
                self.train[k]=copy.deepcopy(trainlist)
                self.val[k]=copy.deepcopy(vallist)
        print("Creating training and validation datasets.") 
        
##LOGISTICS
###############################################################################
        
    def err(self): 
        """
        Calculates mean square error (mse) between self.current and self.train['resp'].
        This is specifically to be used in the basic_min function for model fitting.
        
        Function returns the mse
        """
        E=0
        P=0
        mse=0
        #if tile==True:
            #reps=self.shapes[1]
            #tiled=np.tile(self.current,(1,reps,1))
            #E+=np.sum(np.square(tiled-self.train['resp']))
            #P=np.sum(np.square(self.train['resp']))
        #else:
        E+=np.sum(np.square(self.current-self.train['resp']))
        P=np.sum(np.square(self.train['resp']))
        mse=E/P
        self.mse=mse
        return(mse)
    
    
    def fit_to_phi(self,to_fit):
        """
        Converts fit parameters to a single vector, to be used in fitting
        algorithms.
        to_fit should be formatted ['par1','par2',] etc
        """
        phi=[]
        for k in to_fit:
            g=getattr(self,k).flatten()
            phi=np.append(phi,g)
        return(phi)
    
    
    def phi_to_fit(self,phi,to_fit):
        """
        Converts single fit vector back to fit parameters so model can be calculated
        on fit update steps.
        to_fit should be formatted ['par1','par2',] etc
        """
        st=0
        for k in to_fit:
            s=getattr(self,k).shape
            setattr(self,k,phi[st:(st+np.prod(s))].reshape(s))
            st+=np.prod(s)
     

    def normalize_0to1(data):
        """
        Normalizes data on a 0 to 1 range, just normalizing over time and trial for now.
        Currently DEPRECATED, this may change in the future.
        Input data should be formatted as (Time,Repetition,Stimulus), and should be a numpy array
        """
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
           
    
##FITTERS##
###############################################################################


    def basic_min(self,functions,routine='L-BFGS-B',maxit=50000):
        """
        The basic fitting routine used to fit a model. This function defines a cost
        function that evaluates the functions being fit using the current parameters
        for those functions and outputs the current mean square error (mse). 
        This cost function is evaulated by the scipy optimize.minimize routine,
        which seeks to minimize the mse by changing the function parameters. 
        This function has err, fit_to_phi, and phi_to_fit as dependencies.
        
        Scipy optimize.minimize is set to use the minimization algorithm 'L-BFGS-B'
        as a default, as this is memory efficient for large parameter counts and seems
        to produce good results. However, there are many other potential algorithms 
        detailed in the documentation for optimize.minimize
        
        Function returns self.pred, the current model prediction. 
        """
        params=[]
        for i in functions:
            params=params+self.fit_param[i]
        def cost_fn(phi):
            self.phi_to_fit(phi,to_fit=params)
            for f in functions:
                getattr(self.impdict[f],f)(self,indata=self.train['stim'],
                       data=self.current,pupdata=self.train['pup'],pred=self.pred)
                #TODO: remove indata step so that any function can be first in the queue
            mse=self.err()
            cost_fn.counter+=1
            if cost_fn.counter % 1000==0:
                print('Eval #'+str(cost_fn.counter))
                print('MSE='+str(mse))
            return(mse)
        opt=dict.fromkeys(['maxiter'])
        opt['maxiter']=int(maxit)
        #if function=='tanhON':
            #cons=({'type':'ineq','fun':lambda x:np.array([x[0]-0.01,x[1]-0.01,-x[2]-1])})
            #routine='COBYLA'
        #else:
            #
        cons=()
        phi0=self.fit_to_phi(to_fit=params) 
        cost_fn.counter=0
        sp.optimize.minimize(cost_fn,phi0,method=routine,
                                    constraints=cons,options=opt)
        for f in functions:
                self.pred=getattr(self.impdict[f],f)(self,indata=self.train['stim'],
                       data=self.current,pupdata=self.train['pup'],pred=self.pred)
        print(self.mse)
        return(self.pred)
    
    #TODO: def basinhopping_min
    
    """    
    LOOK INTO scikit-learn FOR POSSIBLY BETTER REGRESSION FUNCTIONS
    """
    
    
        
##OUTPUT MODULES##
###############################################################################

    

    def pupil_comparison(self):
        """
        This method compares the effects of a pupil gain module on a "perfect" model created
        by averaging over all trials for a stimulus.
        
        Returns the ratio of the fit mse calculated with pupil gain and without pupil gain. 
        It the ratio >1, it indicates that the fit with pupil gian fit better. 
        This function is currently NOT USEABLE
        """
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

        
    
    def run_fit(self,validation=0.05,reps=1,save=False,filepath=None):
        """
        This method fits a model for a single instance of the FERReT object. It
        performs all the relevant data setup, including creating a simple validation
        and training data set from the data imported when FERReT was instantiated.
        
        reps specifies the number of rounds of fitting to perform. Higher repetitions
        results in better fits, but with diminishing returns. The method can also 
        save the fitted parameters to a file if desired.
        
        This method has data_resample, reshape_repetitions, resp_avg, create_datasets,
        and basic_min as dependencies. 
        """
        #TODO: remove indata step so that any function can be first in the queue
        self.data_resample(noise_thresh=0.04)
        if self.data['pup'] is not None:
            self.reshape_repetitions()
        else:
            self.resp_avg()
        self.create_datasets(valsize=validation)
        ii=self.queue.index(self.model)
        for i in range(0,reps):
            print('Repetition #'+str(i+1))
            self.basic_min(functions=self.queue[:ii+1])
            for j in self.queue[ii+1:]:
                self.basic_min(functions=[j])
            print(self.mse) 
        params=dict()
        for i,j in self.fit_param.items():
            params[j[0]]=getattr(self,j[0])
        if save is True:
            np.save(filepath,params)
            
        
    
    def apply_to_val(self,save=False,filepath=None):
        """
        This method applies the fitted parameters to the validation data set stimuli
        to produce predicted PSTH's. It wraps the predictions in a dictionary along 
        with the validation stimuli, response, and pupil data.
        
        The function can save this dictionary to disk if desired. the filename should 
        end in a .npy. The dictionary will be wrapped in a numpy array.
        
        Returns a dictionary of numpy arrays, with keys 'stim', 'resp', 'pup', and
        'predicted'.
        """
        def shape_back(ins,origdim):
            s=ins.shape
            outs=np.reshape(ins,(origdim[0],origdim[1],s[1]),order='F')
            return(outs)
        dats=dict.fromkeys(['stim','resp','pup','predicted'])
        for f in self.queue:
            pred=getattr(self.impdict[f],f)(self,indata=self.val['stim'],
                        data=self.current,pupdata=self.val['pup'],pred=self.current)
        if self.data['pup'] is not None:
            dats['predicted']=shape_back(pred,origdim=self.shapes)
            for i,j in self.val.items():
                if i in ('pup','resp'):   
                    dats[i]=shape_back(j,origdim=self.shapes)
                else:
                    s=self.shapes
                    dats[i]=j[:,:s[0],:]
        else:
            dats['predicted']=pred
            for i,j in self.val.items():
                if i in('resp','stim'):
                    dats[i]=j
        if save is True:
            np.save(filepath,dats)
        return(dats)
    
    
    def apply_to_train(self,save=False,filepath=None):
        """
        This method applies the fitted parameters to the training data set stimuli
        to produce predicted PSTH's. It wraps the predictions in a dictionary along 
        with the training stimuli, response, and pupil data.
        
        The function can save this dictionary to disk if desired. the filename should 
        end in a .npy. The dictionary will be wrapped in a numpy array.
        
        Returns a dictionary of numpy arrays, with keys 'stim', 'resp', 'pup', and
        'predicted'.
        """
        def shape_back(ins,origdim):
            s=ins.shape
            outs=np.reshape(ins,(origdim[0],origdim[1],s[1]),order='F')
            return(outs)
        dats=dict.fromkeys(['stim','resp','pup','predicted'])
        for f in self.queue:
            pred=getattr(self.impdict[f],f)(self,indata=self.train['stim'],
                        data=self.current,pupdata=self.train['pup'],pred=self.current)
        if self.data['pup'] is not None:
            dats['predicted']=shape_back(pred,origdim=self.shapes)
            for i,j in self.train.items():
                if i in ('pup','resp'):   
                    dats[i]=shape_back(j,origdim=self.shapes)
                else:
                    s=self.shapes
                    dats[i]=j[:,:s[0],:]
        else:
            dats['predicted']=pred
            for i,j in self.train.items():
                if i in('resp','stim'):
                    dats[i]=j
        if save is True:
            np.save(filepath,dats)
        return(dats)
    
    
    
    
    
    
    
        
        
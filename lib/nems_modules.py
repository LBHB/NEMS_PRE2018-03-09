import numpy as np
import matplotlib.pyplot as plt, mpld3 #mpld3 alias needed for quick_plot_save
import scipy.io
import scipy.signal as sps
import scipy.stats as spstats
import copy
import lib.nems_utils as nu
import math as mt
import scipy.special as sx

class nems_module:
    """nems_module

    Generic NEMS module

    """
    
    #
    # common attributes for all modules
    #
    name='pass-through'
    user_editable_fields=['input_name','output_name']
    plot_fns=[nu.plot_spectrogram]
    
    input_name='stim'  # name of input matrix in d_in
    output_name='stim' # name of output matrix in d_out
    parent_stack=None # pointer to stack instance that owns this module
    id=None  # unique name for this module to be referenced from the stack??
    d_in=None  # pointer to input of data stack, ie, for modules[i], parent_stack.d[i]
    d_out=None # pointer to output, parent_stack.d[i+!]
    fit_fields=[]  # what fields should be fed to phi for fitting

    #
    # Begin standard functions
    #
    def __init__(self,parent_stack=None,**xargs):
        """
        __init__
        Standard initialization for all modules. Sets up next step in data
        stream linking parent_stack.data to self.d_in and self.d_out.
        Also configures default plotter and calls self.my_init(), which can 
        optionally be defined to perform module-specific initialization.
        """
        print("creating module "+self.name)
        
        if parent_stack is None:
            self.d_in=[]
        else:
            # point to parent in order to allow access to it attributes
            self.parent_stack=parent_stack
            # d_in is by default the last entry of parent_stack.data
            self.d_in=parent_stack.data[-1]
            self.id="{0}{1}".format(self.name,len(parent_stack.modules))
        
        self.d_out=copy.copy(self.d_in)
        self.do_plot=self.plot_fns[0]  # default is first in list
        self.do_trial_plot=self.plot_fns[0]
        self.my_init(**xargs)
        
    def parms2phi(self):
        """
        parms2phi - extract all parameter values contained in properties
        listed in self.fit_fields so that they can be passed to fit routines.
        """
        phi=np.empty(shape=[0,1])
        for k in self.fit_fields:
            phi=np.append(phi,getattr(self, k).flatten())
        return phi
        
    def phi2parms(self,phi=[]):
        """
        phi2parms - import fit parameter values from a vector provided by a 
        fit routine
        """
        os=0;
        #print(phi)
        for k in self.fit_fields:
            s=getattr(self, k).shape
            #phi=np.array(phi)
            setattr(self,k,phi[os:(os+np.prod(s))].reshape(s))
            os+=np.prod(s)
    
    def unpack_data(self,name='stim',est=True):
        """
        unpack_data - extract a data variable from all files into a single
        matrix (concatenated across files)
        """
        m=self
        if m.d_in[0][name].ndim==2:
            X=np.empty([0,1])
        else:
            s=m.d_in[0][name].shape
            X=np.empty([s[0],0])
            
        for i, d in enumerate(m.d_in):
            if not 'est' in d.keys():
                if d[name].ndim==2:
                    X=np.concatenate((X,d[name].reshape([-1,1],order='C')))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1],order='C')),axis=1)
            elif (est and d['est']):
                if d[name].ndim==2:
                    X=np.concatenate((X,d[name].reshape([-1,1],order='C')))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1],order='C')),axis=1)
            elif not est and not d['est']:
                if d[name].ndim==2:
                    X=np.concatenate((X,d[name].reshape([-1,1],order='C')))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1],order='C')),axis=1)
                
        return X
    
    def evaluate(self):
        """
        evaluate - iterate through each file, extracting the input data 
        (X=self.data_in[self.input_name]) and passing it as matrix to 
        self.my_eval(), which can perform the module-specific
        transformation (default is a simple pass-through) output of my_eval
        is saved to self.d_out[self.output_name].
        Notice that a deepcopy is made of the input variable so that changes
        to it will not accidentally propagate backwards through the data
        stream
        """
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
        
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in[self.input_name])
            f_out[self.output_name]=self.my_eval(X)
    
    #
    # customizable functions
    #
    def my_init(self,**xargs):
        # placeholder for module specific initialization
        pass 
        
    def my_eval(self,X):
        # placeholder for module-specific evaluation, default is
        # pass-through of pointer to input data matrix.
        Y=X
        return Y
      
    #def do_plot(self,size=(12,4),idx=None):
    #    #deprecated. plot functions are now in nems_utils.py
    #   
    #    # Moved from pylab to pyplot module in all do_plot functions, changed plots 
    #    #to be individual large figures, added other small details -njs June 16, 2017
    #    if idx:
    #        plt.figure(num=idx,figsize=size)
    #    out1=self.d_out[:][self.parent_stack.plot_dataidx]
    #
    #    if out1['stim'].ndim==3:
    #        plt.imshow(out1['stim'][:,self.parent_stack.plot_stimidx,:], aspect='auto', origin='lower')
    #    elif out1['pupil'] is None:
    #    #else:
    #        s=out1['stim'][self.parent_stack.plot_stimidx,:]
    #        r=out1['resp'][self.parent_stack.plot_stimidx,:]
    #        pred, =plt.plot(s,label='Predicted')
    #        resp, =plt.plot(r,'r',label='Response')
    #        plt.legend(handles=[pred,resp])
    #
    #    
    #    else:
    #     u=0
    #        c=out1['repcount'][self.parent_stack.plot_stimidx]
    #        h=out1['stim'][self.parent_stack.plot_stimidx].shape
    #        scl=h[1]/c
    #        
    #        for i in self.parent_stack.plot_trialidx:
    #            s=out1['stim'][self.parent_stack.plot_stimidx,u:(u+scl)]
    #            r=out1['resp'][self.parent_stack.plot_stimidx,u:(u+scl)]
    #            pred, =plt.plot(s,label='Predicted')
    #            resp, =plt.plot(r,'r',label='Response')
    #            plt.legend(handles=[pred,resp])
    #            u=u+scl
        
                
                
    #    plt.title("{0} (data={1}, stim={2})".format(self.name,self.parent_stack.plot_dataidx,self.parent_stack.plot_stimidx))
        
    
#### end nems_module ##########################################################

"""
Data loader modules, typically first entry in the stack
"""

class dummy_data(nems_module):
    """
    dummy_data - generate some very dumb test data without loading any files
    """
    name='dummy_data'
    user_editable_fields=['output_name','data_len']
    plot_fns=[nu.plot_spectrogram]
    data_len=100
    
    def my_init(self,data_len=100):
        self.data_len=data_len

    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
        
        self.d_out[0][self.output_name]=np.zeros([12,2,self.data_len])
        self.d_out[0][self.output_name][0,0,10:19]=1
        self.d_out[0][self.output_name][0,0,30:49]=1
        self.d_out[0]['resp']=self.d_out[0]['stim'][0,:,:]*2+1        
        self.d_out[0]['repcount']=np.sum(np.isnan(self.d_out[0]['resp'])==False,axis=0)

class load_mat(nems_module):
    """
    avg_resp tells load_mat to average the response raster trials. Don't use with pupil 
    data.
    
    perfect_model tells load_mat to use the averaged response raster as the stimulus. Use
    this to just look at pupil effects without fitting the rest of a model.
    """
    name='load_mat'
    user_editable_fields=['output_name','est_files','fs']
    plot_fns=[nu.plot_spectrogram, nu.plot_spectrogram]
    do_trial_plot=plot_fns[0]
    est_files=[]
    fs=100
    
    def my_init(self,est_files=[],fs=100,avg_resp=True,perfect_model=False):
        self.est_files=est_files.copy()
        self.do_trial_plot=self.plot_fns[0]
        self.fs=fs
        self.avg_resp=avg_resp
        self.perfect_model=perfect_model

    def evaluate(self):
        del self.d_out[:]
#        for i, d in enumerate(self.d_in):
#            self.d_out.append(d.copy())
                    
        # load contents of Matlab data file
        for f in self.est_files:
            #f='tor_data_por073b-b1.mat'
            matdata = scipy.io.loadmat(f,chars_as_strings=True)
            for s in matdata['data'][0]:
                try:
                    data={}
                    data['resp']=s['resp_raster']
                    data['stim']=s['stim']
                    data['respFs']=s['respfs'][0][0]
                    data['stimFs']=s['stimfs'][0][0]
                    data['stimparam']=[str(''.join(letter)) for letter in s['fn_param']]
                    data['isolation']=s['isolation']
                    data['prestim']=s['tags'][0]['PreStimSilence'][0][0][0]
                    data['poststim']=s['tags'][0]['PostStimSilence'][0][0][0]
                    data['duration']=s['tags'][0]['Duration'][0][0][0] 
                except:
                    data = scipy.io.loadmat(f,chars_as_strings=True)
                    data['raw_stim']=data['stim'].copy()
                    data['raw_resp']=data['resp'].copy()
                try:
                    data['pupil']=s['pupil']
                except:
                    data['pupil']=None
                try:
                    if s['estfile']:
                        data['est']=True
                    else:
                        data['est']=False
                except:
                    print("Est/val conditions not flagged in datafile")
                    
    #            data = scipy.io.loadmat(f,chars_as_strings=True)
    #            data['raw_stim']=data['stim'].copy()
    #            data['raw_resp']=data['resp'].copy()
                                    
                data['fs']=self.fs
                noise_thresh=0.04
                stim_resamp_factor=int(data['stimFs']/self.fs)
                resp_resamp_factor=int(data['respFs']/self.fs)
                
                self.parent_stack.unresampled={'resp':data['resp'],'respFs':data['respFs'],'duration':data['duration'],
                                               'poststim':data['poststim'],'prestim':data['prestim'],'pupil':data['pupil']}
                
                
                # reshape stimulus to be channel X time
                data['stim']=np.transpose(data['stim'],(0,2,1))
                if stim_resamp_factor != 1:
                    s=data['stim'].shape
                    #new_stim_size=np.round(s[2]*stim_resamp_factor)
                    #print('resampling stim from '+str(data['stimFs'])+'Hz to '+str(self.fs)+'Hz.')
                    resamp=sps.decimate(data['stim'],stim_resamp_factor,ftype='fir',axis=2,zero_phase=True)
                    s_indices=resamp<noise_thresh
                    resamp[s_indices]=0
                    data['stim']=resamp
                    #data['stim']=scipy.signal.resample(data['stim'],new_stim_size,axis=2)
                    
                # resp time (axis 0) should be resampled to match stim time (axis 1)
                if resp_resamp_factor != 1:
                    s=data['resp'].shape
                    #new_resp_size=np.round(s[0]*resp_resamp_factor)
                    #print('resampling resp from '+str(data['respFs'])+'Hz to '+str(self.fs)+'Hz.')
                    resamp=sps.decimate(data['resp'],resp_resamp_factor,ftype='fir',axis=0,zero_phase=True)
                    s_indices=resamp<noise_thresh
                    resamp[s_indices]=0
                    data['resp']=resamp
                    #data['resp']=scipy.signal.resample(data['resp'],new_resp_size,axis=0)
                    
                if data['pupil'] is not None and resp_resamp_factor != 1:
                    s=data['pupil'].shape
                    #new_resp_size=np.round(s[0]*resp_resamp_factor)
                    #print('resampling pupil from '+str(data['respFs'])+'Hz to '+str(self.fs)+'Hz.')
                    resamp=sps.decimate(data['pupil'],resp_resamp_factor,ftype='fir',axis=0,zero_phase=True)
                    s_indices=resamp<noise_thresh
                    resamp[s_indices]=0
                    data['pupil']=resamp
                    #data['pupil']=scipy.signal.resample(data['pupil'],new_resp_size,axis=0)
                    
                #Changed resample to decimate w/ 'fir' and threshold, as it produces less ringing when downsampling
                #-njs June 16, 2017
                    
                # average across trials
                #TODO: need to fix repcount for data with varying trial numbers (in pupil_est_val)
                data['repcount']=np.sum(np.isnan(data['resp'][0,:,:])==False,axis=0)
                self.parent_stack.unresampled['repcount']=data['repcount']
                #print(data['stim'].shape)
                #print(data['resp'].shape)
                #print(data['pupil'].shape)
                

                if self.avg_resp is True: 
                    data['resp']=np.nanmean(data['resp'],axis=1) 
                    #data['resp']=np.transpose(data['resp'],(1,0))
                elif self.perfect_model is True:
                    avg=np.nanmean(data['resp'],axis=1)
                    data['stim']=avg
                    
                    #else:
                        #sr=data['resp'].shape
                        #data['stim']=np.tile(data['stim'],(1,sr[1],1))
                        #print(data['stim'].shape)
                    #for i in ('resp','pupil'):
                        #s=data[i].shape
                        #print(s)
                        #data[i]=np.reshape(data[i],(s[0],s[1]*s[2]),order='F')
                        #data[i]=np.transpose(data[i],(1,0))
                        #print(data[i].shape)


                    
                # append contents of file to data, assuming data is a dictionary
                # with entries stim, resp, etc...
                print('load_mat: appending {0} to d_out stack'.format(f))
                self.d_out.append(data)

            
            # each trial is (PreStimSilence + Duration + PostStimSilence) sec long
            #Duration=data['Duration'][0,0] # Duration of TORC sounds
            #PreStimSilence=data['PreStimSilence'][0,0]
            #PostStimSilence=data['PostStimSilence'][0,0]

    

class standard_est_val(nems_module):
    """
    Special module(s) for organizing/splitting estimation and validation data.
    Currently just one that replicates (mostly) the standard procedure from NARF
    """
 
    name='standard_est_val'
    user_editable_fields=['output_name','valfrac']
    valfrac=0.05
    
    def my_init(self, valfrac=0.05):
        self.valfrac=valfrac
    
    def evaluate(self):
        del self.d_out[:]
         # for each data file:
        for i, d in enumerate(self.d_in):
            #self.d_out.append(d)
            try:
                if d['est']:
                    # flagged as est data
                    self.d_out.append(d)
                elif self.parent_stack.valmode:
                    self.d_out.append(d)
                    
            except:
                # est/val not flagged, need to figure out
                
                #--made a new est/val specifically for pupil --njs, June 28 2017
                
                # figure out number of distinct stim
                s=d['repcount']
                
                m=s.max()
                validx = s==m
                estidx = s<m
                
                d_est=d.copy()
                d_val=d.copy()
                
                d_est['repcount']=copy.deepcopy(d['repcount'][estidx])
                d_est['resp']=copy.deepcopy(d['resp'][estidx,:])
                d_est['stim']=copy.deepcopy(d['stim'][:,estidx,:])
                d_val['repcount']=copy.deepcopy(d['repcount'][validx])
                d_val['resp']=copy.deepcopy(d['resp'][validx,:])
                d_val['stim']=copy.deepcopy(d['stim'][:,validx,:])
                
                #if 'pupil' in d.keys():
                if d['pupil'] is not None:
                    d_est['pupil']=copy.deepcopy(d['pupil'][estidx,:])
                    d_val['pupil']=copy.deepcopy(d['pupil'][validx,:])
                    #for j in (d_est,d_val):
                    #    for i in ('resp','pupil'):
                    #        s=j[i].shape
                    #        j[i]=np.reshape(j[i],(s[0]*s[1],s[2]),order='F')
                    #        j[i]=np.transpose(j[i],(1,0))
                    #    j['stim']=np.tile(j['stim'],(1,1,s[1]))
                
                d_est['est']=True
                d_val['est']=False
                
                self.d_out.append(d_est)
                if self.parent_stack.valmode:
                    self.d_out.append(d_val)
                    
class stim_est_val(nems_module):
    """
    DEPRECATED, use crossval with use_trials=False
    
    Use with datasets that have large numbers of stimuli with few trials
    """
    
    name='stim_est_val'
    user_editable_fields=['output_name','valfrac']
    valfrac=0.05
    
    def my_init(self,valfrac=0.05):
        self.valfrac=valfrac
        self.crossval=self.parent_stack.cross_val
        try:
            self.iter=int(1/valfrac)-1
        except:
            self.iter=0
            
    def evaluate(self):
        del self.d_out[:]
         # for each data file:
        for i, d in enumerate(self.d_in):
            re=d['resp'].shape
            spl=mt.ceil(re[0]*self.valfrac)
            
            count=self.parent_stack.cv_counter
            count=count*spl
            
            d_est=d.copy()
            d_val=d.copy()
            
            d_val['repcount']=copy.deepcopy(d['repcount'][count:(count+spl)])
            d_val['resp']=copy.deepcopy(d['resp'][count:(count+spl),:])
            d_val['pupil']=copy.deepcopy(d['pupil'][count:(count+spl),:])
            d_val['stim']=copy.deepcopy(d['stim'][:,count:(count+spl),:])
            
            d_est['repcount']=np.delete(d['repcount'],np.s_[count:(count+spl)],0)
            d_est['resp']=np.delete(d['resp'],np.s_[count:(count+spl)],0)
            d_est['pupil']=np.delete(d['pupil'],np.s_[count:(count+spl)],0)
            d_est['stim']=np.delete(d['stim'],np.s_[count:(count+spl)],1)

            
            d_est['est']=True
            d_val['est']=False
                 
            self.d_out.append(d_est)
            if self.parent_stack.valmode:
                self.d_out.append(d_val)
                
        if self.parent_stack.cv_counter==self.iter:
            self.parent_stack.cond=True


class trial_est_val(nems_module):
    """
    DEPRECATED, use crossval with use_trials=True
    
    Breaks imported data into est/val. Use with pupil_model and batch 294 data, where 
    there are only 2 stimuli, so takes some trials from both stimuli as validation data.
    This module is specfically for looking at just pupil gain, without other 
    model fitting.
    
    Use with batch 294-like data sets with high numbers of trials but low number of stimuli.
    
    Compatible with cross-validation. 
    """
 
    name='pupil_est_val'
    user_editable_fields=['output_name','valfrac']
    #plot_fns=[nu.non_plot]
    valfrac=0.05
    
    def my_init(self, valfrac=0.05):
        self.valfrac=valfrac
        self.crossval=self.parent_stack.cross_val
        try:
            self.iter=int(1/valfrac)-1
        except:
            self.iter=0
    
    def evaluate(self):
        del self.d_out[:]
         # for each data file:
        for i, d in enumerate(self.d_in):
            re=d['resp'].shape
            respl=mt.ceil(re[1]*(1-self.valfrac))
            spl=mt.ceil(re[1]*self.valfrac)
            
            count=self.parent_stack.cv_counter
            count=count*spl

            #print('count='+str(count))
            d_est=d.copy()
            d_val=d.copy()
            
            #TODO: need to fixed repcounts for batches with different trial numbers for different stims
            #TODO: want to create a way to delete using mask for varying trial numbers
            d_val['repcount']=np.full(shape=re[2],fill_value=spl,dtype='int64') 
            d_val['resp']=copy.deepcopy(d['resp'][:,count:(count+spl),:])
            d_val['stim']=copy.deepcopy(d['stim'])
            d_val['pupil']=copy.deepcopy(d['pupil'][:,count:(count+spl),:])
            d_est['repcount']=np.full(shape=re[2],fill_value=respl,dtype='int64')
            d_est['resp']=np.delete(d['resp'],np.s_[count:(count+spl)],1)
            d_est['stim']=copy.deepcopy(d['stim'])
            d_est['pupil']=np.delete(d['pupil'],np.s_[count:(count+spl)],1)

            #d_est['repcount']=copy.deepcopy(d['repcount'][:respl])
            #d_est['resp']=copy.deepcopy(d['resp'][:,:respl,:])
            ##d_est['stim']=copy.deepcopy(d['stim'][:,:stspl,:])
            ##d_val['stim']=copy.deepcopy(d['stim'][:,stspl:,:])

            #d_est['pupil']=copy.deepcopy(d['pupil'][:,:respl,:])
            #d_val['pupil']=copy.deepcopy(d['pupil'][:,respl:,:])
                
            d_est['est']=True
            d_val['est']=False
            
            self.d_out.append(d_est)
            if self.parent_stack.valmode:
                self.d_out.append(d_val)
                
        if self.parent_stack.cv_counter==self.iter:
            self.parent_stack.cond=True
            
class crossval(nems_module):
    """
    Cross-validation est/val module that replaces trial_est_val and stim_est_val.
    """
    name='crossval'
    valfrac=0.05
    
    def my_init(self,valfrac=0.05,use_trials=False):
        self.valfrac=valfrac
        self.crossval=self.parent_stack.cross_val
        self.use_trials=use_trials
        try:
            self.iter=int(1/valfrac)-1
        except:
            self.iter=0
        
    def evaluate(self):
        del self.d_out[:]
         # for each data file:
        for i, d in enumerate(self.d_in):    
            count=self.parent_stack.cv_counter
        
            d_est=d.copy()
            d_val=d.copy()
            
            if self.use_trials is True:
                re=d['resp'].shape
                respl=mt.ceil(re[1]*(1-self.valfrac))
                spl=mt.ceil(re[1]*self.valfrac)
                count=count*spl
                print('Creating estimation/validation datasets using trials')
                try:
                    d_val['pupil']=copy.deepcopy(d['pupil'][:,count:(count+spl),:])
                    d_est['pupil']=np.delete(d['pupil'],np.s_[count:(count+spl)],1)
                except TypeError:
                    print('No pupil data')
                    d_val['pupil']=None
                    d_est['pupil']=None
                d_val['resp']=copy.deepcopy(d['resp'][:,count:(count+spl),:])
                d_est['resp']=np.delete(d['resp'],np.s_[count:(count+spl)],1)
                d_val['stim']=copy.deepcopy(d['stim'])
                d_est['stim']=copy.deepcopy(d['stim'])
                d_val['repcount']=np.full(shape=re[2],fill_value=spl,dtype='int64')
                d_est['repcount']=np.full(shape=re[2],fill_value=respl,dtype='int64')
            else:
                re=d['resp'].shape
                spl=mt.ceil(re[-1]*self.valfrac)
                count=count*spl
                print('Creating estimation/validation datasets using stimuli')
                try:
                    d_val['pupil']=copy.deepcopy(d['pupil'][:,:,count:(count+spl)])
                    d_est['pupil']=np.delete(d['pupil'],np.s_[count:(count+spl)],2)
                except TypeError:
                    print('No pupil data')
                    d_val['pupil']=None
                    d_est['pupil']=None
                try:
                    d_val['resp']=copy.deepcopy(d['resp'][:,:,count:(count+spl)])
                    d_est['resp']=np.delete(d['resp'],np.s_[count:(count+spl)],2)
                except IndexError:
                    d['resp']=np.transpose(d['resp'],(1,0))
                    d_val['resp']=copy.deepcopy(d['resp'][count:(count+spl),:])
                    d_est['resp']=np.delete(d['resp'],np.s_[count:(count+spl)],0)
                try:
                    d_val['stim']=copy.deepcopy(d['stim'][:,count:(count+spl),:])
                    d_est['stim']=np.delete(d['stim'],np.s_[count:(count+spl)],1)
                except IndexError:
                    d_val['stim']=copy.deepcopy(d['stim'][:,count:(count+spl)])
                    d_est['stim']=np.delete(d['stim'],np.s_[count:(count+spl)],1)
                d_val['repcount']=copy.deepcopy(d['repcount'][count:(count+spl)])
                d_est['repcount']=np.delete(d['repcount'],np.s_[count:(count+spl)],0)
                    
            d_est['est']=True
            d_val['est']=False
            
            self.d_out.append(d_est)
            if self.parent_stack.valmode:
                self.d_out.append(d_val)
                
        if self.parent_stack.cv_counter==self.iter:
            self.parent_stack.cond=True
            
                    
class pupil_model(nems_module):
    name='pupil_model'
    plot_fns=[nu.sorted_raster,nu.raster_plot]
    """
    Just reshapes & tiles stim, resp, and pupil data correctly for looking at pupil gain.
    Will probably incorporate into pupil_est_val later.
    """
    def my_init(self,tile_data=True):
        self.tile_data=tile_data
   
    def evaluate(self):
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            self.d_out.append(val.copy())
            self.d_out[-1][self.output_name]=copy.deepcopy(self.d_out[-1][self.output_name])
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in['resp'])
            Xp=copy.deepcopy(f_in['pupil'])
            #Xa=np.nanmean(X,axis=1)
            Xa=copy.deepcopy(f_in['stim'])
            if self.tile_data is True:
                s=Xp.shape 
                #Z=np.reshape(Xp,(s[0]*s[1],s[2]),order='F') #Uncomment to have long "stimuli"
                Z=np.reshape(Xp,(s[0],s[1]*s[2]),order='F')  #Comment out to have long "stimuli"
                Z=np.transpose(Z,(1,0))
                #Q=np.reshape(X,(s[0]*s[1],s[2]),order='F') #Uncomment to have long "stimuli"
                Q=np.reshape(X,(s[0],s[1]*s[2]),order='F') #Comment out to have long "stimuli"
                Q=np.transpose(Q,(1,0))
                #Y=np.tile(Xa,(s[1],1)) #Uncomment to have long "stimuli"
                Y=np.tile(Xa[:,0],(s[1],1))
                for i in range(1,Xa.shape[1]):
                    Y=np.append(Y,np.tile(Xa[:,i],(s[1],1)),axis=0) #Comment out to have long "stimuli"  
                #Y=np.transpose(Y,(1,0))
            else:
                Y=X
            f_out[self.output_name]=Y  
            f_out['pupil']=Z
            f_out['resp']=Q
            

"""
Modules that actually transform the data stream
"""

class normalize(nems_module):
    """
    normalize - rescale a variable, typically stim, to put it in a range that
    works well with fit algorithms --
    either mean 0, variance 1 (if sign doesn't matter) or
    min 0, max 1 (if positive values desired)
    IMPORTANT NOTE: normalization factors are computed from estimation data 
    only but applied to both estimation and validation data streams
    """
    name='normalize'
    user_editable_fields=['output_name','valfrac','valmode']
    force_positive=True
    d=0
    g=1
    
    def my_init(self, force_positive=True,data='stim'):
        self.force_positive=force_positive
        self.input_name=data
    
    def evaluate(self):
        X=self.unpack_data()
        name=self.input_name
        
        if self.d_in[0][name].ndim==2:
            if self.force_positive:
                self.d=X.min()
                self.g=1/(X-self.d).max()
            else:
                self.d=X.mean()
                self.g=X.std()
        else:
            s=self.d_in[0][name].shape
            if self.force_positive:
                self.d=X[:,:].min(axis=1).reshape([s[0],1,1])
                self.g=1/(X[:,:]-self.d.reshape([s[0],1])).max(axis=1).reshape([s[0],1,1])
            else:
                self.d=X[:,:].mean(axis=1).reshape([s[0],1,1])
                self.g=X[:,:].std(axis=1).reshape([s[0],1,1])
                self.g[np.isinf(g)]=0
                
        # apply the normalization
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
        
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in[self.input_name])
            f_out[self.output_name]=np.multiply(X-self.d,self.g)
                        
       
class add_scalar(nems_module):
    """ 
    add_scalar -- pretty much a dummy test module but may be useful for
    some reason
    """
    name='add_scalar'
    user_editable_fields=['output_name','n']
    n=np.zeros([1,1])
    
    def my_init(self, n=0, fit_fields=['n']):
        self.fit_fields=fit_fields
        self.n[0,0]=n
                   
    def my_eval(self,X):
        Y=X+self.n
        return Y
    
class dc_gain(nems_module):
    """ 
    dc_gain -- apply a scale and offset term
    """
 
    name='dc_gain'
    user_editable_fields=['output_name','d','g']
    d=np.zeros([1,1])
    g=np.ones([1,1])
    
    def my_init(self, d=0, g=1, fit_fields=['d','g']):
        self.fit_fields=fit_fields
        self.d[0,0]=d
        self.g[0,0]=g
    
    def my_eval(self,X):
        Y=X*self.g+self.d
        return Y
   
        
class sum_dim(nems_module):
    """
    sum_dim - sum a matrix across one dimension. maybe useful? mostly testing
    """
    name='sum_dim'
    user_editable_fields=['output_name','dim']
    dim=0
    
    def my_init(self, dim=0):
        self.dim=dim
        
    def my_eval(self,X):
        Y=X.sum(axis=self.dim)
        return Y
            
       
class weight_channels(nems_module):
    """
    weight_channels - apply a weighting matrix across a variable in the data
    stream. Used to provide spectral filters, directly imported from NARF.
    a helper function parm_fun can be defined to parameterized the weighting
    matrix. but by default the weights are each independent
    """
    name='weight_channels'
    user_editable_fields=['output_name','num_dims','coefs','baseline','phi','parm_fun']
    plot_fns=[nu.plot_strf,nu.plot_spectrogram]
    coefs=None
    baseline=np.zeros([1,1])
    num_chans=1
    parm_fun=None
    
    def my_init(self, num_dims=0, num_chans=1, baseline=np.zeros([1,1]), 
                fit_fields=['coefs'], parm_fun=None, phi=np.zeros([1,1])):
        if self.d_in and not(num_dims):
            num_dims=self.d_in[0]['stim'].shape[0]
        self.num_dims=num_dims
        self.num_chans=num_chans
        self.baseline=baseline
        self.fit_fields=fit_fields
        if parm_fun:
            self.parm_fun=parm_fun
            self.coefs=parm_fun(phi)
        else:
            #self.coefs=np.ones([num_chans,num_dims])/num_dims/100
            self.coefs=np.random.normal(1,0.1,[num_chans,num_dims])/num_dims
        self.phi=phi
        
    def my_eval(self,X):
        #if not self.d_out:
        #    # only allocate memory once, the first time evaling. rish is that output_name could change
        if self.parm_fun:
            self.coefs=self.parm_fun(self.phi)
        s=X.shape
        X=np.reshape(X,[s[0],-1])
        X=np.matmul(self.coefs,X)
        s=list(s)
        s[0]=self.num_chans
        Y=np.reshape(X,s)
        return Y
    
 
class fir_filter(nems_module):
    """
    fir_filter - the workhorse linear filter module
    """
    name='fir_filter'
    user_editable_fields=['output_name','num_dims','coefs','baseline']
    plot_fns=[nu.plot_strf, nu.plot_spectrogram]
    coefs=None
    baseline=np.zeros([1,1])
    num_dims=0
    
    def my_init(self, num_dims=0, num_coefs=20, baseline=0, fit_fields=['baseline','coefs']):
        if self.d_in and not(num_dims):
            num_dims=self.d_in[0]['stim'].shape[0]
        self.num_dims=num_dims
        self.num_coefs=num_coefs
        self.baseline[0]=baseline
        self.coefs=np.zeros([num_dims,num_coefs])
        self.fit_fields=fit_fields
        self.do_trial_plot=self.plot_fns[0]
        
    def my_eval(self,X):
        #if not self.d_out:
        #    # only allocate memory once, the first time evaling. rish is that output_name could change
        s=X.shape
        X=np.reshape(X,[s[0],-1])
        for i in range(0,s[0]):
            y=np.convolve(X[i,:],self.coefs[i,:])
            X[i,:]=y[0:X.shape[1]]
        X=X.sum(0)+self.baseline
        Y=np.reshape(X,s[1:])
        return Y



class nonlinearity(nems_module): 
    """
    nonlinearity - apply a static nonlinearity. TODO: use helper functions
    rather than a look-up table to determine which NL to apply. parameters can
    be saved in a generic vector self.phi - see NARF implementation for reference 
    
    @author: shofer
    """
    #Added helper functions and removed look up table --njs June 29 2017
    name='nonlinearity'
    plot_fns=[nu.pre_post_psth,nu.plot_spectrogram]
    user_editable_fields = ['nltype', 'fit_fields','phi']
    phi=np.array([1])
    
    def my_init(self,d_in=None,my_eval=None,nltype='dlog',fit_fields=['phi'],phi=[1],premodel=False):
        if premodel is True:
            self.do_plot=self.plot_fns[2]
        self.fit_fields=fit_fields
        self.nltype=nltype
        self.phi=np.array([phi])
        #setattr(self,nltype,phi0)
        if my_eval is None:
            if nltype=='dlog':
                self.my_eval=self.dlog_fn
                self.plot_fns=[nu.plot_spectrogram]
                self.do_plot=self.plot_fns[0]
            elif nltype=='exp':
                self.my_eval=self.exp_fn
            elif nltype=='dexp':
                self.my_eval=self.dexp_fn
        else:
            self.my_eval=my_eval
            
        
    #TODO: could even put these functions in a separate module?
    def dlog_fn(self,X):
        Y=np.log(X+self.phi[0,0])
        return(Y)
    def exp_fn(self,X):
        Y=np.exp(self.phi[0,0]*(X-self.phi[0,1]))
        return(Y)
    def dexp_fn(self,X):
        Y=self.phi[0,0]-self.phi[0,1]*np.exp(-np.exp(self.phi[0,2]*(X-self.phi[0,3])))
        return(Y)
    def poly_fn(self,X):
        deg=self.phi.shape[1]
        Y=0
        for i in range(0,deg):
            Y+=self.phi[0,i]*np.power(X,i)
        return(Y)
    def tanh_fn(self,X):
        Y=self.phi[0,0]*np.tanh(self.phi[0,1]*X-self.phi[0,2])+self.phi[0,0]
        return(Y)
        
    def my_eval(self,X):
        Z=getattr(self,self.nltype+'_fn')(X)
        return(Z)

                 
class state_gain(nems_module): 
    """
    state_gain - apply a gain/offset based on continuous pupil diameter, or some other continuous variable.
    my not be able to use standard my_eval() because needs access to two 
    variables in the data stream rather than just one.
    
    @author: shofer
    """
    #Changed to helper function based general module --njs June 29 2017
    name='state_gain'
    plot_fns=[nu.pre_post_psth,nu.non_plot]
    
    def my_init(self,d_in=None,gain_type='linpupgain',fit_fields=['theta'],theta=[0,1,0,0],premodel=False,
                order=None):
        if premodel is True:
            self.do_plot=self.plot_fns[1]
        #self.linpupgain=np.zeros([1,4])
        #self.linpupgain[0][1]=0
        self.fit_fields=fit_fields
        self.gain_type=gain_type
        theta=np.array([theta])
        self.theta=theta
        self.order=order
        self.do_plot=self.plot_fns[0]
        #self.data_setup(d_in)
        print('state_gain parameters created')
        
    def nopupgain_fn(self,X,Xp):
        """
        Applies a simple dc gain & offset to the stim data. Does not actually involve 
        state variable. This is the "control" for the state_gain exploration.
        """
        Y=self.theta[0,0]+self.theta[0,1]*X
        return(Y)   
    def linpupgain_fn(self,X,Xp):
        Y=self.theta[0,0]+(self.theta[0,2]*Xp)+(self.theta[0,1]*X)+self.theta[0,3]*np.multiply(Xp,X)
        return(Y)
    def exppupgain_fn(self,X,Xp):
        Y=self.theta[0,0]+self.theta[0,1]*X*np.exp(self.theta[0,2]*Xp+self.theta[0,3])
        return(Y)
    def logpupgain_fn(self,X,Xp):
        Y=self.theta[0,0]+self.theta[0,1]*X*np.log(self.theta[0,2]+Xp+self.theta[0,3])
        return(Y)
    def polypupgain_fn(self,X,Xp):
        """
        Fits a polynomial gain function: 
        Y = g0 + g*X + d1*X*Xp^1 + d2*X*Xp^2 + ... + d(n-1)*X*Xp^(n-1) + dn*X*Xp^n
        """
        deg=self.theta.shape[1]
        Y=0
        for i in range(0,deg-2):
            Y+=self.theta[0,i]*X*np.power(Xp,i+1)
        Y+=self.theta[0,-2]+self.theta[0,-1]*X
        return(Y)
    def powerpupgain_fn(self,X,Xp):
        """
        Slightly different than polypugain. Y = g0 + g*X + d0*Xp^n + d*X*Xp^n
        """
        deg=self.order
        v=self.theta
        Y=v[0,0] + v[0,1]*X + v[0,2]*np.power(Xp,deg) + v[0,3]*np.multiply(X,np.power(Xp,deg))
        return(Y)
    def Poissonpupgain_fn(self,X,Xp): #Kinda useless, might delete ---njs
        u=self.theta[0,1]
        Y=self.theta[0,0]*X*np.divide(np.exp(-u)*np.power(u,Xp),sx.factorial(Xp))
        return(Y)
    def butterworthHP_fn(self,X,Xp):
        """
        Applies a Butterworth high pass filter to the pupil data, with a DC offset.
        Pupil diameter is treated here as analogous to frequency, and the fitted 
        parameters are DC offset, overall gain, and f3dB. Order is specified, and
        controls how fast the rolloff is.
        """
        n=self.order
        Y=self.theta[0,2]+self.theta[0,0]*X*np.divide(np.power(np.divide(Xp,self.theta[0,1]),n),
                    np.sqrt(1+np.power(np.divide(Xp,self.theta[0,1]),2*n)))
        return(Y)
    
    
    def evaluate(self):
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            #self.d_out.append(copy.deepcopy(val))
            self.d_out.append(val.copy())
            self.d_out[-1][self.output_name]=copy.deepcopy(self.d_out[-1][self.output_name])        
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in[self.input_name])
            Xp=copy.deepcopy(f_in['pupil'])
            Z=getattr(self,self.gain_type+'_fn')(X,Xp)
            f_out[self.output_name]=Z
        
"""
modules for computing scores/ assessing model performance
"""
class mean_square_error(nems_module):
 
    name='mean_square_error'
    user_editable_fields=['input1','input2','norm']
    plot_fns=[nu.pred_act_psth,nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    norm=True
    shrink=0
    mse_est=np.ones([1,1])
    mse_val=np.ones([1,1])
        
    def my_init(self, input1='stim',input2='resp',norm=True,shrink=0):
        self.input1=input1
        self.input2=input2
        self.norm=norm
        self.shrink=shrink
        self.do_trial_plot=self.plot_fns[1]
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
            
        if self.shrink:
            X1=self.unpack_data(self.input1,est=True)
            X2=self.unpack_data(self.input2,est=True)
            bounds=np.round(np.linspace(0,len(X1)+1,11)).astype(int)
            E=np.zeros([10,1])
            P=np.mean(np.square(X2))
            for ii in range(0,10):
                E[ii]=np.mean(np.square(X1[bounds[ii]:bounds[ii+1]]-X2[bounds[ii]:bounds[ii+1]]))
            E=E/P
            mE=E.mean()
            sE=E.std()
            if mE<1:
                # apply shrinkage filter to 1-E with factors self.shrink
                mse=1-nu.shrinkage(1-mE,sE,self.shrink)
            else:
                mse=mE
                
        else:
            E=np.zeros([1,1])
            P=np.zeros([1,1])
            N=0
            for f in self.d_out:
                #E+=np.sum(np.sum(np.sum(np.square(f[self.input1]-f[self.input2]))))
                E+=np.sum(np.square(f[self.input1]-f[self.input2]))
                #P+=np.sum(np.sum(np.sum(np.square(f[self.input2]))))
                P+=np.sum(np.square(f[self.input2]))
                N+=f[self.input2].size
        
            if self.norm:
                mse=E/P
            else:
                mse=E/N
                
        if self.parent_stack.valmode is True:   
            self.mse_val=mse
            self.parent_stack.meta['mse_val']=mse
        else:
            self.mse_est=mse
            self.parent_stack.meta['mse_est']=mse
        
        
        return mse

    def error(self, est=True):
        if est:
            return self.mse_est
        else:
            # placeholder for something that can distinguish between est and val
            return self.mse_val
        
class pseudo_huber_error(nems_module):
    """
    Pseudo-huber "error" to use with fitter cost functions. This is more robust to
    ouliers than simple mean square error. Approximates L1 error at large
    values of error, and MSE at low error values. Has the additional benefit (unlike L1)
    of being convex and differentiable at all places.
    
    Pseudo-huber equation taken from Hartley & Zimmerman, "Multiple View Geometry
    in Computer Vision," (Cambridge University Press, 2003), p619
    
    C(delta)=2(b^2)(sqrt(1+(delta/b)^2)-1)
    
    b mediates the value of error at which the the error is penalized linearly or quadratically.
    Note that setting b=1 is the soft l1 loss
    
    @author: shofer, June 30 2017
    """
    #I think this is working (but I'm not positive). When fitting with a pseudo-huber
    #cost function, the fitter tends to ignore areas of high spike rates, but does
    #a good job finding the mean spike rate at different times during a stimulus. 
    #This makes sense in that the huber error penalizes outliers, and could be 
    #potentially useful, depending on what is being fit? --njs, June 30 2017
    
    
    name='pseudo_huber_error'
    plot_fns=[nu.pred_act_psth,nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    b=0.9 #sets the value of error where fall-off goes from linear to quadratic\
    huber_est=np.ones([1,1])
    huber_val=np.ones([1,1])
    
    def my_init(self, input1='stim',input2='resp',b=0.9):
        self.input1=input1
        self.input2=input2
        self.b=b
        self.do_trial_plot=self.plot_fns[1]
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
    
        for f in self.d_out:
            delta=np.divide(np.sum(f[self.input1]-f[self.input2],axis=1),np.sum(f[self.input2],axis=1))
            C=np.sum(2*np.square(self.b)*(np.sqrt(1+np.square(np.divide(delta,self.b)))-1))
            C=np.array([C])
        self.huber_est=C
            
    def error(self,est=True):
        if est is True:
            return(self.huber_est)
        else: 
            return(self.huber_val)
        
            
        
class correlation(nems_module):
 
    name='correlation'
    user_editable_fields=['input1','input2']
    plot_fns=[nu.pred_act_psth, nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    r_est=np.ones([1,1])
    r_val=np.ones([1,1])
        
    def my_init(self, input1='stim',input2='resp',norm=True):
        self.input1=input1
        self.input2=input2
        self.do_plot=self.plot_fns[1]
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())

        X1=self.unpack_data(self.input1,est=True)            
        X2=self.unpack_data(self.input2,est=True)
        r_est,p=spstats.pearsonr(X1,X2)
        self.r_est=r_est
        self.parent_stack.meta['r_est']=r_est
                              
        X1=self.unpack_data(self.input1,est=False)            
        if X1.size:
            X2=self.unpack_data(self.input2,est=False)
            r_val,p=spstats.pearsonr(X1,X2)
            self.r_val=r_val
            self.parent_stack.meta['r_val']=r_val
        
            return r_val
        else:
            return (r_est)
    
        

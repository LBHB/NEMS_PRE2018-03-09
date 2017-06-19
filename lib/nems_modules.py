import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal as sps
import scipy.stats as spstats
import copy
import lib.nems_utils as nu

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
        self.my_init(**xargs)
        
    def parms2phi(self):
        phi=np.empty(shape=[0,1])
        for k in self.fit_fields:
            phi=np.append(phi,getattr(self, k).flatten())
        return phi
        
    def phi2parms(self,phi=[]):
        os=0;
        for k in self.fit_fields:
            s=getattr(self, k).shape
            setattr(self,k,phi[os:(os+np.prod(s))].reshape(s))
            os+=np.prod(s)
    
    def unpack_data(self,name='stim',est=True):
        m=self
        if m.d_in[0][name].ndim==2:
            X=np.empty([0,1])
        else:
            s=m.d_in[0][name].shape
            X=np.empty([s[0],0])
            
        for i, d in enumerate(m.d_in):
            if est and d['est']:
                if d[name].ndim==2:
                    X=np.concatenate((X,d[name].reshape([-1,1])))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1])),axis=1)
            if not est and not d['est']:
                if d[name].ndim==2:
                    X=np.concatenate((X,d[name].reshape([-1,1])))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1])),axis=1)
                
        return X
    
    def evaluate(self):
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
        # initialization specfic to this module
        0 # null op?
        
    def my_eval(X):
        # default: pass-through pointers to data from input to output,
        # need to deepcopy individual dict entries if they are changed
        Y=X
        return Y
        
    def do_plot(self,size=(12,4),idx=None):
        #Moved from pylab to pyplot module in all do_plot functions, changed plots 
        #to be individual large figures, added other small details -njs June 16, 2017
        if idx:
            plt.figure(num=idx,figsize=size)
        out1=self.d_out[:][self.parent_stack.plot_dataidx]
        if out1['stim'].ndim==3:
            plt.imshow(out1['stim'][:,self.parent_stack.plot_stimidx,:], aspect='auto', origin='lower')
        else:
            s=out1['stim'][self.parent_stack.plot_stimidx,:]
            r=out1['resp'][self.parent_stack.plot_stimidx,:]
            pred, =plt.plot(s,label='Predicted')
            resp, =plt.plot(r,'r',label='Response')
            plt.legend(handles=[pred,resp])
                
        plt.title("{0} (data={1}, stim={2})".format(self.name,self.parent_stack.plot_dataidx,self.parent_stack.plot_stimidx))
            
        
# end nems_module

class dummy_data(nems_module):

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

    name='load_mat'
    user_editable_fields=['output_name','est_files','fs']
    plot_fns=[nu.plot_spectrogram]
    est_files=[]
    fs=100
    
    def my_init(self,est_files=[],fs=100):
        self.est_files=est_files.copy()
        self.fs=fs

    def evaluate(self):
        del self.d_out[:]
#        for i, d in enumerate(self.d_in):
#            self.d_out.append(d.copy())
                    
        # load contents of Matlab data file
        for f in self.est_files:
            #f='tor_data_por073b-b1.mat'
            matdata = scipy.io.loadmat(f,chars_as_strings=True)
            s=matdata['data'][0][0]
            try:
                data={}
                data['resp']=s['resp_raster']
                data['stim']=s['stim']
                data['respFs']=s['respfs'][0][0]
                print(data['respFs'])
                data['stimFs']=s['stimfs'][0][0]
                print(data['stimFs'])
                data['stimparam']=[str(''.join(letter)) for letter in s['fn_param']]
                data['isolation']=s['isolation']
            except:
                data = scipy.io.loadmat(f,chars_as_strings=True)
                data['raw_stim']=data['stim'].copy()
                data['raw_resp']=data['resp'].copy()
            try:
                data['pupil']=s['pupil']
            except:
                data['pupil']=None
#            data = scipy.io.loadmat(f,chars_as_strings=True)
#            data['raw_stim']=data['stim'].copy()
#            data['raw_resp']=data['resp'].copy()
                
            data['fs']=self.fs
            noise_thresh=0.04
            stim_resamp_factor=int(data['stimFs']/self.fs)
            resp_resamp_factor=int(data['respFs']/self.fs)
            
            
            # reshape stimulus to be channel X time
            data['stim']=np.transpose(data['stim'],(0,2,1))
            if stim_resamp_factor != 1:
                s=data['stim'].shape
                #new_stim_size=np.round(s[2]*stim_resamp_factor)
                print('resampling stim from '+str(data['stimFs'])+'Hz to '+str(self.fs)+'Hz.')
                resamp=sps.decimate(data['stim'],stim_resamp_factor,ftype='fir',axis=2,zero_phase=True)
                s_indices=resamp<noise_thresh
                resamp[s_indices]=0
                data['stim']=resamp
                #data['stim']=scipy.signal.resample(data['stim'],new_stim_size,axis=2)
                
            # resp time (axis 0) should be resampled to match stim time (axis 1)
            if resp_resamp_factor != 1:
                s=data['resp'].shape
                #new_resp_size=np.round(s[0]*resp_resamp_factor)
                print('resampling resp from '+str(data['respFs'])+'Hz to '+str(self.fs)+'Hz.')
                resamp=sps.decimate(data['resp'],resp_resamp_factor,ftype='fir',axis=0,zero_phase=True)
                s_indices=resamp<noise_thresh
                resamp[s_indices]=0
                data['resp']=resamp
                #data['resp']=scipy.signal.resample(data['resp'],new_resp_size,axis=0)
                
            if data['pupil'] is not None and resp_resamp_factor != 1:
                s=data['pupil'].shape
                #new_resp_size=np.round(s[0]*resp_resamp_factor)
                print('resampling pupil from '+str(data['respFs'])+'Hz to '+str(self.fs)+'Hz.')
                resamp=sps.decimate(data['pupil'],resp_resamp_factor,ftype='fir',axis=0,zero_phase=True)
                s_indices=resamp<noise_thresh
                resamp[s_indices]=0
                data['pupil']=resamp
                #data['pupil']=scipy.signal.resample(data['pupil'],new_resp_size,axis=0)
                
            #Changed resmaple to decimate w/ 'fir' and threshold, as it produces less ringing when downsampling
            #-njs June 16, 2017
                
            # average across trials
            print(np.isnan(data['resp'][0,:,:]).shape)
            data['repcount']=np.sum(np.isnan(data['resp'][0,:,:])==False,axis=0)
            
            #TODO: remove avging & add trial reshape if there is pupil data
            if data['pupil'] is None:
                data['resp']=np.nanmean(data['resp'],axis=1) 
                data['resp']=np.transpose(data['resp'],(1,0))
            
            # append contents of file to data, assuming data is a dictionary
            # with entries stim, resp, etc...
            print('load_mat: appending {0} to d_out stack'.format(f))
            self.d_out.append(data)

            # spectrogram of TORC stimuli. 15 frequency bins X 300 time samples X 30 different TORCs
            #stim=data['stim']
            #FrequencyBins=data['FrequencyBins'][0,:]
            #stimFs=data['stimFs'][0,0]
            #StimCyclesPerSec=data['StimCyclesPerSec'][0,0]
            #StimCyclesPerSec=np.float(StimCyclesPerSec)
            
            # response matrix. sampled at 1kHz. value of 1 means a spike occured
            # in a particular time bin. 0 means no spike. shape: [3000 time bins X 2
            # repetitions X 30 different TORCs]
                                                                  
            #resp=data['resp']
            #respFs=data['respFs'][0,0]
            
            # each trial is (PreStimSilence + Duration + PostStimSilence) sec long
            #Duration=data['Duration'][0,0] # Duration of TORC sounds
            #PreStimSilence=data['PreStimSilence'][0,0]
            #PostStimSilence=data['PostStimSilence'][0,0]

class standard_est_val(nems_module):
 
    name='standard_est_val'
    user_editable_fields=['output_name','valfrac','valmode']
    valfrac=0.05
    
    def my_init(self, valfrac=0.05):
        self.valfrac=valfrac
    
    def evaluate(self):
        del self.d_out[:]
         # for each data file:
        for i, d in enumerate(self.d_in):
            #self.d_out.append(d)
            
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
            
            d_est['est']=True
            d_val['est']=False
            
            self.d_out.append(d_est)
            if self.parent_stack.valmode:
                self.d_out.append(d_val)

        
       
class add_scalar(nems_module):
 
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

    name='sum_dim'
    user_editable_fields=['output_name','dim']
    dim=0
    
    def my_init(self, dim=0):
        self.dim=dim
        
    def my_eval(self,X):
        Y=X.sum(axis=self.dim)
        return Y
            
    
class fir_filter(nems_module):
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
    

class dexp(nems_module):
    
    name='dexp'
    user_editable_fields=['output_name','dexp']
    plot_fns=[nu.pred_act_psth, nu.pred_act_scatter]
    dexp=np.ones([1,4])
        
    def my_init(self,dexp=np.ones([1,4]),fit_fields=['dexp']):
        self.dexp=dexp 
        self.fit_fields=fit_fields

    def my_eval(self,X):
        v1=self.dexp[0,0]
        v2=self.dexp[0,1]
        v3=self.dexp[0,2]
        v4=self.dexp[0,3]
        Y=v1-v2*np.exp(-np.exp(v3*(X-v4)))
        return Y
    
#    def do_plot(self,size=(12,4),idx=None):
#        #if ax is None:
#            #pl.set_cmap('jet')
#            #pl.figure()
#            #ax=pl.subplot(1,1,1)
#            
#        if idx:
#            plt.figure(num=idx,figsize=size)
#        in1=self.d_in[self.parent_stack.plot_dataidx]
#        out1=self.d_out[self.parent_stack.plot_dataidx]
#        s1=in1['stim'][self.parent_stack.plot_stimidx,:]
#        s2=out1['stim'][self.parent_stack.plot_stimidx,:]
#        pre, =plt.plot(s1,label='Pre-nonlinearity')
#        post, =plt.plot(s2,'r',label='Post-nonlinearity')
#        plt.legend(handles=[pre,post])
#        plt.title("{0} (data={1}, stim={2})".format(self.name,self.parent_stack.plot_dataidx,self.parent_stack.plot_stimidx))
        
class nonlinearity(nems_module): 
    
    name='nonlinearity'
    
    def __init__(self,d_in=None,nltype='dlog',fit_fields=['dlog']):
        self.nltype=nltype
        self.fit_fields=fit_fields
        self.data_setup(d_in)
        if nltype=='dlog':
            self.dlog=np.ones([1,1])
        elif nltype=='exp':
            self.exp=np.ones([1,2])
            self.exp[0][1]=0
        #etc...
        
        
    def evaluate(self):
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            #self.d_out.append(copy.deepcopy(val))
            self.d_out.append(val.copy())
            self.d_out[-1][self.output_name]=copy.deepcopy(self.d_out[-1][self.output_name])
        
        if self.nltype=='dlog':
            v1=self.dlog[0,0]
            for f_in,f_out in zip(self.d_in,self.d_out):
                X=copy.deepcopy(f_in[self.input_name])
                X=np.log(X+v1)
                f_out[self.output_name]=X
        elif self.nltype=='exp':
            v1=self.exp[0,0]
            v2=self.exp[0,1]
            for f_in,f_out in zip(self.d_in,self.d_out):
                X=copy.deepcopy(f_in[self.input_name])
                X=np.exp(v1*(X-v2))
                f_out[self.output_name]=X
        #etc...
        
        
#    def do_plot(self,size=(12,4),idx=None):
#        print('No nonlinearity plot yet')
            
            
        
        
        
        

#TODO: finish linpupgain/figure out best way to load in pupil data 
class linpupgain(nems_module): 
    
    name='linpupgain'
    
    def __init__(self,d_in=None,fit_fields=['linpugain']):
        self.linpupgain=np.zeros([1,4])
        self.linpupgain[0][1]=0
        self.fit_fields=fit_fields
        self.data_setup(d_in)
        print('linpupgain parameters created')     
    
    def evaluate(self):
        d0=self.linpupgain[0,0]
        g0=self.linpupgain[0,1]
        d=self.linpupgain[0,2]
        g=self.linpupgain[0,3]
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            #self.d_out.append(copy.deepcopy(val))
            self.d_out.append(val.copy())
            self.d_out[-1][self.output_name]=copy.deepcopy(self.d_out[-1][self.output_name])        
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in[self.input_name])
            X=v1-v2*np.exp(-np.exp(v3*(X-v4)))
            f_out[self.output_name]=X
        
        output=d0+(d*pups)+(g0*ins)+g*np.multiply(pups,ins)
 
        

class mean_square_error(nems_module):
 
    name='mean_square_error'
    user_editable_fields=['input1','input2','norm']
    plot_fns=[nu.pred_act_psth, nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    norm=True
    mse_est=np.ones([1,1])
    mse_val=np.ones([1,1])
        
    def my_init(self, input1='stim',input2='resp',norm=True):
        self.input1=input1
        self.input2=input2
        self.norm=norm
        
    def evaluate(self):
        del self.d_out[:]
        for i, d in enumerate(self.d_in):
            self.d_out.append(d.copy())
            
        E=np.zeros([1,1])
        P=np.zeros([1,1])
        N=0
        for f in self.d_out:
            E+=np.sum(np.sum(np.sum(np.square(f[self.input1]-f[self.input2]))))
            P+=np.sum(np.sum(np.sum(np.square(f[self.input2]))))
            N+=f[self.input2].size
    
        if self.norm:
            mse=E/P
        else:
            mse=E/N
        self.mse_est=mse
        self.parent_stack.meta['mse_est']=mse
        
        return mse

    def error(self, est=True):
        if est:
            return self.mse_est
        else:
            # placeholder for something that can distinguish between est and val
            return self.mse_val
        
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
            return r_est
    
        
class nems_stack:
        
    """nems_stack

    Properties:
     modules = list of nems_modules in sequence of execution

    """
    modelname=None
    modules=[]  # stack of modules
    mod_names=[]
    mod_ids=[]
    data=[]     # corresponding stack of data in/out for each module
    meta={}
    fitter=None
    valmode=False
    
    plot_dataidx=0
    plot_stimidx=0
    
    def __init__(self):
        print("Creating new stack")
        self.modules=[]
        self.mod_names=[]
        self.data=[]
        self.data.append([])
        self.data[0].append({})
        self.data[0][0]['resp']=[]
        self.data[0][0]['stim']=[]
        
        self.meta={}
        self.modelname='Empty stack'
        self.error=self.default_error
        self.valmode=False
        
    def evaluate(self,start=0):
        # evalute stack, starting at module # start
        for ii in range(start,len(self.modules)):
            #if ii>0:
            #    print("Propagating mod {0} d_out to mod{1} d_in".format(ii-1,ii))
            #    self.modules[ii].d_in=self.modules[ii-1].d_out
            self.modules[ii].evaluate()
    
    # create instance of mod and append to stack    
    def append(self, mod=None, **xargs):
        if mod is None:
            m=nems_module(self)
        else:
            m=mod(self, **xargs)
        
        self.modules.append(m)
        self.data.append(m.d_out)
        self.mod_names.append(m.name)
        self.mod_ids.append(m.id)
        m.evaluate()
        
    def popmodule(self, mod=nems_module()):
        del self.modules[-1]
        del self.data[-1]
        
    def output(self):
        return self.data[-1]
    
    def default_error(self):
        return np.zeros([1,1])
    
    def quick_plot(self):
        plt.figure(figsize=(8,9))
        for idx,m in enumerate(self.modules):
            # skip first module
            if idx>0:
                plt.subplot(len(self.modules)-1,1,idx)
                m.do_plot(m)
            
# end nems_stack



        

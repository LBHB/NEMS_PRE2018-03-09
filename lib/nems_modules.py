import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal as sps
import copy

empty_data={}

class nems_data:
    """nems_data

    NOT USED CURRENTLY!  JUST USING LIST OF DICTIONARIES FOR DATA STACK!
    
    Generic NEMS data bucket

    provides input and output of each nems_module

    structure containing a set of matrices, corresponding to input(s)
    and output(s). eg, resp, stim, stim2, state, etc.

    """
    est_files=[]
    val_files=[]
    data=[]
    
    def __init__(self):
        self.data=[]

    def d(self,n=None):
        if n is None:
            return self.data
        else:
            return self.data[n]

    def copy_keys(self,d_in=None):
        if d_in != None:
            self.est_files=d_in.est_files
            self.val_files=d_in.val_files
            self.data=d_in.data.copy()

# end nems_data

        
        
class nems_module:
    """nems_module

    Generic NEMS module

    """
    
    # common properties for all modules
    name='pass-through'
    input_name='stim'  # name of input matrix in d_in
    output_name='stim' # name of output matrix in d_out
    phi=None # vector of parameter values that can be fit
    d_in=None
    d_out=None
    fit_params=[]
    meta={}

    def __init__(self,d_in=None):
        self.data_setup(d_in)
        
    def data_setup(self,d_in=None):
        if d_in is None:
            self.d_in=[] # list of data buckets fed into module
        else:
            self.d_in=d_in
        self.d_out=[] # list of outputs, same size as data in

    def prep_eval(self):
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            self.d_out.append(val)
        
    def evaluate(self):
        # default: pass-through pointers to data from input to output,
        # need to deepcopy individual dict entries if they are changed
        self.prep_eval()
        
    def parms2phi(self):
        phi=np.empty(shape=[0,1])
        for k in self.fit_params:
            phi=np.append(phi,getattr(self, k).flatten())
        return phi
        
    def phi2parms(self,phi=[]):
        os=0;
        for k in self.fit_params:
            s=getattr(self, k).shape
            setattr(self,k,phi[os:(os+np.prod(s))].reshape(s))
            os+=np.prod(s)
            
    def do_plot(self,size=(12,4),idx=None):
        if idx:
            plt.figure(num=idx,figsize=size)
        out1=self.d_out[:]
        plt.imshow(out1[0]['stim'][:,0,:], aspect='auto', origin='lower')
        plt.title(self.name)
        
        #Moved from pylab to pyplot module in all do_plot functions, changed plots 
        #to be individual large figures, added other small details -njs June 16, 2017
# end nems_module

class load_mat(nems_module):

    name='load_mat'
    est_files=[]
    val_files=[]
    fs=100
    
    def __init__(self,d_in=None,est_files=[],val_files=[],fs=100):
        self.data_setup(d_in)
        self.est_files=est_files.copy()
        self.val_files=val_files.copy()
        self.fs=fs

    def evaluate(self):
        self.prep_eval()
        self.meta['est_files']=self.est_files
        self.meta['val_files']=self.val_files
        
        # new list object for dat
        del self.d_out[:]
        
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
    valfrac=0.05
    valmode=False
    
    def __init__(self, d_in=None, valfrac=0.05):
        self.valfrac=valfrac
        self.valmode=False
        self.data_setup(d_in)
    
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
            if self.valmode:
                self.d_out.append(d_val)

        
       
class add_scalar(nems_module):
 
    name='add_scalar'
    
    def __init__(self, d_in=None, n=1, fit_params=['n']):
        self.fit_params=fit_params
        self.data_setup(d_in)
        self.n=n
       
    def evaluate(self):
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            self.d_out.append(copy.deepcopy(val))

        for f in self.d_out:
            f[self.output_name]=f[self.input_name]+self.n
        
class sum_dim(nems_module):
    name='sum_dim'
     
    def __init__(self, d_in=None, dim=1):
        self.data_setup(d_in)
        self.dim=dim
        
    def evaluate(self):
        del self.d_out[:]
        # for each data file:
        for i, val in enumerate(self.d_in):
            self.d_out.append(copy.deepcopy(val))

        for f in self.d_out:
            f[self.output_name]=f[self.input_name].sum(axis=self.dim)

            
class fir_filter(nems_module):
    name='fir_filter'
    coefs=None
    baseline=np.zeros([1,1])
    num_dims=0
    
    def __init__(self, d_in=None, num_dims=0, num_coefs=20, baseline=0, fit_params=['baseline','coefs']):
        if d_in and not(num_dims):
            num_dims=d_in[0]['stim'].shape[0]
        self.num_dims=num_dims
        self.num_coefs=num_coefs
        self.baseline[0]=baseline
        self.coefs=np.zeros([num_dims,num_coefs])
        
        self.fit_params=fit_params
        self.data_setup(d_in)
        
    def evaluate(self):
        #if not self.d_out:
        #    # only allocate memory once, the first time evaling. rish is that output_name could change
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            #self.d_out.append(copy.deepcopy(val))
            self.d_out.append(val.copy())
            #self.d_out[-1][self.output_name]=copy.deepcopy(self.d_out[-1][self.output_name])
            
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in[self.input_name])
            s=X.shape
            X=np.reshape(X,[s[0],-1])
            for i in range(0,s[0]):
                y=np.convolve(X[i,:],self.coefs[i,:])
                X[i,:]=y[0:X.shape[1]]
            X=X.sum(0)+self.baseline
            f_out[self.output_name]=np.reshape(X,s[1:])
    
    def do_plot(self,size=(12,4),idx=None):
        #if ax is None:
            #pl.set_cmap('jet')
            #pl.figure()
            #ax=pl.subplot(1,1,1)
        
        if idx:
            plt.figure(num=idx,figsize=size)
        h=self.coefs
        plt.imshow(h, aspect='auto', origin='lower',cmap=plt.get_cmap('jet'))
        plt.colorbar()
        plt.title(self.name)

class dexp(nems_module):
    
    name='dexp'
    dexp=np.ones([1,4]) 
    
    def __init__(self,d_in=None,dexp=None,fit_params=['dexp']):
        if dexp is None:
            self.dexp=np.ones([1,4]) 
        self.fit_params=fit_params
        self.data_setup(d_in)

    def evaluate(self):
        v1=self.dexp[0,0]
        v2=self.dexp[0,1]
        v3=self.dexp[0,2]
        v4=self.dexp[0,3]
        del self.d_out[:]
        for i, val in enumerate(self.d_in):
            #self.d_out.append(copy.deepcopy(val))
            self.d_out.append(val.copy())
            self.d_out[-1][self.output_name]=copy.deepcopy(self.d_out[-1][self.output_name])
            
        for f_in,f_out in zip(self.d_in,self.d_out):
            X=copy.deepcopy(f_in[self.input_name])
            X=v1-v2*np.exp(-np.exp(v3*(X-v4)))
            f_out[self.output_name]=X
    
    def do_plot(self,size=(12,4),idx=None):
        #if ax is None:
            #pl.set_cmap('jet')
            #pl.figure()
            #ax=pl.subplot(1,1,1)
            
        if idx:
            plt.figure(num=idx,figsize=size)
        in1=self.d_in[0]
        out1=self.d_out[0]
        s1=in1['stim'][0,:]
        s2=out1['stim'][0,:]
        pre, =plt.plot(s1/s1.max(),label='Pre-nonlinearity')
        post, =plt.plot(s2/s2.max(),'r',label='Post-nonlinearity')
        plt.legend(handles=[pre,post])
        plt.title(self.name)
        
class nonlinearity(nems_module): 
    
    name='nonlinearity'
    
    def __init__(self,d_in=None,nltype='dlog',fit_params=['dlog']):
        self.nltype=nltype
        self.fit_params=fit_params
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
    
    def __init__(self,d_in=None,fit_params=['linpugain']):
        self.linpupgain=np.zeros([1,4])
        self.linpupgain[0][1]=0
        self.fit_params=fit_params
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
    input1='stim'
    input2='resp'
    output=np.ones([1,1])
    norm=True
    
    def __init__(self, d_in=None, input1='stim',input2='resp',norm=True):
        self.data_setup(d_in)
        self.input1=input1
        self.input2=input2
        self.norm=norm
        
    def evaluate(self):
        self.prep_eval()
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
        self.meta['est_mse']=mse
        
        return mse

    def error(self, est_data=True):
        if est_data:
            return self.meta['est_mse']
        else:
            # placeholder for something that can distinguish between est and val
            return self.meta['val_mse']
            
    def do_plot(self,size=(12,4),idx=None):
        if idx:
            plt.figure(num=idx,figsize=size)
        out1=self.d_out[0]
        s=out1['stim'][0,:]
        r=out1['resp'][0,:]
        pred, =plt.plot(s/s.max(),label='Predicted')
        resp, =plt.plot(r/r.max(),'r',label='Response')
        plt.legend(handles=[pred,resp])
        plt.title(self.name)
        
class nems_stack:
        
    """nems_stack

    Properties:
     modules = list of nems_modules in sequence of execution

    """
    modules=[]  # stack of modules
    mod_names=[]
    data=[]     # corresponding stack of data in/out for each module
    modelname=None
    meta={}
    fitter=None
    
    def __init__(self):
        print("dummy")
        self.modules=[]
        self.mod_names=[]
        self.data=[]
        self.meta={}
        self.modelname='Empty stack'
        self.error=self.default_error
        
    def evaluate(self,start=0):
        # evalute stack, starting at module # start
        for ii in range(start,len(self.modules)):
            #if ii>0:
            #    print("Propagating mod {0} d_out to mod{1} d_in".format(ii-1,ii))
            #    self.modules[ii].d_in=self.modules[ii-1].d_out
            self.modules[ii].evaluate()
            
    def append(self, mod=None):
        if mod is None:
            mod=nems_module()
        mod.meta=self.meta
        if len(self.modules):
            print("Propagating d_out from {0} into new d_in".format(self.modules[-1].name))
            mod.d_in=self.data[-1]
        self.modules.append(mod)
        self.mod_names.append(mod.name)
        self.modules[-1].evaluate()
        self.data.append(mod.d_out)
        
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
            plt.subplot(len(self.modules),1,idx+1)
            m.do_plot()
#        for idx,m in enumerate(self.modules):
#            plt.subplot(len(self.modules),1,idx+1)
#            m.do_plot()
            
# end nems_stack



        

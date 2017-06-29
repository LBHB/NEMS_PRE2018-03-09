import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal as sps
import scipy.stats as spstats
import copy
import lib.nems_utils as nu
import math as mt

class nems_module:
    """nems_module

    Generic NEMS module

    """
    
    #
    # common attributes for all modules
    #
    name='pass-through'
    user_editable_fields=['input_name','output_name']
    plot_fns=[nu.plot_spectrogram,nu.plot_trials]
    
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
        unpack_data - extact a data variable from all files into a single
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
                    X=np.concatenate((X,d[name].reshape([-1,1])))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1])),axis=1)
            elif (est and d['est']):
                if d[name].ndim==2:
                    X=np.concatenate((X,d[name].reshape([-1,1])))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1])),axis=1)
            elif not est and not d['est']:
                if d[name].ndim==2:
                    X=np.concatenate((X,d[name].reshape([-1,1])))
                else:
                    X=np.concatenate((X,d[name].reshape([s[0],-1])),axis=1)
                
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
        
    
# end nems_module

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

    name='load_mat'
    user_editable_fields=['output_name','est_files','fs']
    plot_fns=[nu.plot_spectrogram, nu.plot_spectrogram]
    do_trial_plot=plot_fns[0]
    est_files=[]
    fs=100
    
    def my_init(self,est_files=[],fs=100,formpup=True):
        self.est_files=est_files.copy()
        self.do_trial_plot=self.plot_fns[0]
        self.fs=fs
        self.formpup=formpup

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
                
                self.parent_stack.unresampled=[data['resp'],data['respFs'],data['duration'],
                                               data['poststim'],data['prestim'],data['pupil']]
                
                
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
                
                #print(data['stim'].shape)
                #print(data['resp'].shape)
                #print(data['pupil'].shape)
                
                if data['pupil'] is None: 
                    data['resp']=np.nanmean(data['resp'],axis=1) 
                    data['resp']=np.transpose(data['resp'],(1,0))
                elif data['pupil'] is not None and self.formpup is True:
                    for i in ('resp','pupil'):
                        s=data[i].shape
                        data[i]=np.reshape(data[i],(s[0]*s[1],s[2]),order='F')
                        data[i]=np.transpose(data[i],(1,0))
                    data['stim']=np.tile(data['stim'],(1,1,s[1]))
               #else:
                    #for i in ('resp','pupil'):
                        #data[i]=np.transpose(data[i],(1,0))

                    
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

    
"""
Special module(s) for organizing/splitting estimation and validation data.
Currently just one that replicates (mostly) the standard procedure from NARF
"""
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
                    
class pupil_est_val(nems_module):
    """
    Breaks imported data into est/val. Needed for batch 294-like data, where there is no est/val data
    flagged and low numbers of stimuli
    """
 
    name='pupil_est_val'
    user_editable_fields=['output_name','valfrac','valmode']
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
                st=d['stim'].shape
                re=d['resp'].shape
                stspl=mt.ceil(st[1]*(1-self.valfrac))
                respl=mt.ceil(re[0]*(1-self.valfrac))
                
                
                d_est=d.copy()
                d_val=d.copy()
                

                d_est['repcount']=copy.deepcopy(d['repcount'][:respl])
                d_est['resp']=copy.deepcopy(d['resp'][:respl,:])
                d_est['stim']=copy.deepcopy(d['stim'][:,:stspl,:])
                d_val['repcount']=copy.deepcopy(d['repcount'][respl:])
                d_val['resp']=copy.deepcopy(d['resp'][respl:,:])
                d_val['stim']=copy.deepcopy(d['stim'][:,stspl:,:])
                
                #if 'pupil' in d.keys():
                if d['pupil'] is not None:
                    d_est['pupil']=copy.deepcopy(d['pupil'][:respl,:])
                    d_val['pupil']=copy.deepcopy(d['pupil'][respl:,:])
                
                d_est['est']=True
                d_val['est']=False
                
                self.d_out.append(d_est)
                if self.parent_stack.valmode:
                    self.d_out.append(d_val)
                    
                    
class pupil_model(nems_module):
    name='pupil_model'
    
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
            Xa=np.nanmean(X,axis=1)
            print(X.shape)
            print(Xp.shape)
            if self.tile_data is True:
                s=Xp.shape
                Z=np.reshape(Xp,(s[0]*s[1],s[2]),order='F')
                Z=np.transpose(Z,(1,0))
                Q=np.reshape(X,(s[0]*s[1],s[2]),order='F')
                Q=np.transpose(Q,(1,0))
                Y=np.tile(Xa,(s[1],1))
                Y=np.transpose(Y,(1,0))
                print(Y.shape)
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
    name='standard_est_val'
    user_editable_fields=['output_name','valfrac','valmode']
    force_positive=True
    d=0
    g=1
    
    def my_init(self, force_positive=True):
        self.force_positive=force_positive
    
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
    plot_fns=[nu.plot_strf,nu.plot_trials,nu.plot_spectrogram]
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


#class dexp(nems_module):
    #"""
    #dexp - static sigmoid. TODO : wrapped into the standard static nonlinearity
    #"""
    
    #name='dexp'
    #user_editable_fields=['output_name','dexp']
    #plot_fns=[nu.pred_act_psth, nu.pred_act_scatter]
    #dexp=np.ones([1,4])

        
    #def my_init(self,dexp=np.ones([1,4]),fit_fields=['dexp']):
        #self.dexp=dexp 
        #self.fit_fields=fit_fields

    #def my_eval(self,X):
        #v1=self.dexp[0,0]
        #v2=self.dexp[0,1]
        #v3=self.dexp[0,2]
        #v4=self.dexp[0,3]
        #Y=v1-v2*np.exp(-np.exp(v3*(X-v4)))
        #return Y
    
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

#TODO: make sure that this module is adding parameters to fit?
class nonlinearity(nems_module): 
    """
    nonlinearity - apply a static nonlinearity. TODO: use helper functions
    rather than a look-up table to determine which NL to apply. parameters can
    be saved in a generic vector self.phi - see NARF implementation for reference 
    """
    name='nonlinearity'
    plot_fns=[nu.pre_post_psth,nu.trial_prepost_psth]
    user_editable_fields = ['nltype', 'fit_fields']
    
    
    
    def my_init(self,d_in=None,nltype='dlog',fit_fields=['dlog']):
        #self.nltype=nltype #This might cause an issue if there is more than one nonlinearity...?
        self.fit_fields=fit_fields
        #self.do_trial_plot=self.plot_fns[1]
        self.do_trial_plot=print('no plot yet')
        if nltype=='dlog':
            self.dlog=np.ones([1,1])
        elif nltype=='exp':
            self.exp=np.ones([1,2])
            self.exp[0][1]=0
        elif nltype=='dexp':
            self.dexp=np.ones([1,4])
        #etc...
        
        
    def my_eval(self,X):
        
        #if self.nltype=='dlog':
        if hasattr(self,'dlog'):
            v1=self.dlog[0,0]
            Y=np.log(X+v1)
        #elif self.nltype=='exp':
        elif hasattr(self,'exp'):
            v1=self.exp[0,0]
            v2=self.exp[0,1]
            Y=np.exp(v1*(X-v2))
        #elif self.nltype=='dexp'
        elif hasattr(self,'dexp'):
            v1=self.dexp[0,0]
            v2=self.dexp[0,1]
            v3=self.dexp[0,2]
            v4=self.dexp[0,3]
            Y=v1-v2*np.exp(-np.exp(v3*(X-v4)))
        return Y
        #etc...
        
        return Y        
        
#    def do_plot(self,size=(12,4),idx=None):
#        print('No nonlinearity plot yet')
            
            
        
        
        
        

#TODO: finish linpupgain/figure out best way to load in pupil data 
class linpupgain(nems_module): 
    """
    linpupgain - apply a gain/offset based on pupil diameter, or some other continuous variable.
    my not be able to use standard my_eval() because needs access to two 
    variables in the data stream rather than just one.
    """
    name='linpupgain'
    plot_fns=[nu.trial_prepost_psth]
    
    def my_init(self,d_in=None,fit_fields=['linpupgain']):
        self.linpupgain=np.zeros([1,4])
        self.linpupgain[0][1]=0
        self.fit_fields=fit_fields
        self.do_trial_plot=self.plot_fns[0]
        #self.data_setup(d_in)
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
            Xp=copy.deepcopy(f_in['pupil'])
            Y=d0+(d*Xp)+(g0*X)+g*np.multiply(Xp,X)
            f_out[self.output_name]=Y
        
"""
modules for computing scores/ assessing model performance
"""
class mean_square_error(nems_module):
 
    name='mean_square_error'
    user_editable_fields=['input1','input2','norm']
    plot_fns=[nu.pred_act_psth,nu.plot_trials,nu.pred_act_scatter]
    input1='stim'
    input2='resp'
    norm=True
    mse_est=np.ones([1,1])
    mse_val=np.ones([1,1])
        
    def my_init(self, input1='stim',input2='resp',norm=True):
        self.input1=input1
        self.input2=input2
        self.norm=norm
        self.do_trial_plot=self.plot_fns[1]
        
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
        
    """
    Key components:
     modules = list of nems_modules in sequence of execution
     data = stream of data as it is evaluated through the sequence
            of modules
     fitter = pointer to the fit module
     quick_plot = generates a plot of something about the transformation
                  that takes place at each modules step

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
        self.plot_trialidx=(0,3)
        self.unresampled=[] #If the data is resampled by load_mat, holds an unresampled copy for raster plot
        
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
        
    def append_instance(self, mod=None):
        """Same as append but takes an instance of a module instead
        of the class to preserve existing **xargs. For use with insert/remove.
        Could maybe merge these with an added boolean arg? Wasn't sure if that
        would interfere with the **xargs.
        
        @author: jacob
        
        """
        if not mod:
            mod=nems_module(self)
        self.modules.append(mod)
        self.data.append(mod.d_out)
        self.mod_names.append(mod.name)
        self.mod_ids.append(mod.id)
        mod.evaluate()
        
    def insert(self, mod=None, idx=None, **xargs):
        """Insert a module at index in stack, then evaluate the inserted
        module and re-append all the modules that were in the stack previously,
        starting with the insertion index.
        
        Returns:
        --------
        idx : int
            Index the module was inserted at.
            Will either be the same as the argument idx if given, or
            the last index in the stack if mod was appended instead.
        
        @author: jacob
        
        """
        
        # if no index is given or index is out of bounds,
        # just append mod to the end of the stack
        if (not idx) or (idx > len(self.modules)-1)\
                     or (idx < -1*len(self.modules)-1):
            self.append(mod, **xargs)
            idx = len(self.modules) - 1
            return idx
        
        tail = [self.popmodule_2() for m in self.modules[idx:]]
        self.append(mod, **xargs)
        for mod in reversed(tail[:-1]):
            self.append_instance(mod)
        return idx
    
    def remove(self, idx=None, mod=None, all_idx=False):
        """Remove the module at the given index in stack, then re-append
        all modules that came after the removed module.
        
        Arguments:
        ----------
        idx : int
            Index of module to be removed. If no idx is given, but a mod is
            given, nu.find_modules will be used to find idx.
        mod : nems_module
            Module to be removed. Only needs to be passed if index is not
            given, or if all matching indices should be removed.
        
        all_idx : boolean
            If true, and a module is passed instead without an idx, 
            all matching instances of the module will be removed.
            Otherwise the instance at the highest index will be removed.
            
        Errors:
        -------
        Raises an IndexError if idx is None or its absolute value is greater
        than the length of self.modules, or if mod is given but not found.
        
        @author: jacob
        
        """
        
        if mod and not idx:
            idx = nu.find_modules(self, mod.name)
            if not idx:
                print("Module does not exist in stack.")
                return
            if not all_idx:
                # Only remove the instance with the highest index
                self.remove(idx=idx[-1], mod=None)
                return
            else:
                j = idx[0]
                # same as tail comp below, but exclude all indexes that match
                # the mod
                tail_keep = [
                        self.popmodule_2() for i, m in 
                        enumerate(self.modules[j:])
                        if i not in idx
                        ]
                # still need to pop the matched instances, but don't need to
                # do anything with them.
                tail_toss = [
                        self.popmodule_2() for i, m in
                        enumerate(self.modules[j:])
                        if i in idx
                        ]
                for mod in reversed(tail_keep):
                    self.append_instance(mod)
                        
            
        if (not idx) or (idx > len(self.modules)-1)\
                     or (idx < -1*len(self.modules)-1):
            raise IndexError
        
        # Remove modules from stack starting at the end until reached idx.
        tail = [self.popmodule_2() for m in self.modules[idx:]]
        # Then put them back on starting with the second to last module popped.
        for mod in reversed(tail[:-1]):
            self.append_instance(mod)

        
    def popmodule_2(self):
        """For remove and insert -- wasn't sure if the original popmodule 
        method had a specific use-case, so I didn't want to modify it.
        Can merge the two if these changes won't cause issues.
        
        Removes the last module from stack lists, along with its corresponding
        data and name (and id?).
        
        @author: jacob
        """
        
        m = self.modules.pop(-1)
        self.data.pop(-1)
        self.mod_names.pop(-1)
        # Doesn't look like this one is being used yet?
        #self.mod_ids.pop(-1)
        return m
        
    def popmodule(self, mod=nems_module()):
        del self.modules[-1]
        del self.data[-1]
        
    def output(self):
        return self.data[-1]
    
    def default_error(self):
        return np.zeros([1,1])
    
    def quick_plot(self):
        plt.figure(figsize=(12,15))
        for idx,m in enumerate(self.modules):
            # skip first module
            if idx>0:
                plt.subplot(len(self.modules)-1,1,idx)
                m.do_plot(m)
    
    def trial_quick_plot(self):
        """
        Plots several trials of a stimulus after fitting pupil data.
        This is to make it easier to visualize the fits on individual trials,
        as opposed to over the entire length of the fitted vector.
        """
        
        for idx,m in enumerate(self.modules):
            # skip first module
            if idx>0:
                print('idx= '+str(idx))
                m.do_trial_plot(m,idx)
                
    def do_raster_plot(self):
        """
        Generates a raster plot for the stimulus specified by self.plot_stimidx
        """
        un=self.unresampled
        nu.raster_plot(data=un,stims=self.plot_stimidx,size=(12,6))
        
    def do_sorted_raster(self):
        """
        Generates a raster plot sorted by average pupil diameter for the stimulus
        specified by self.plot_stimidx
        """
        un=copy.deepcopy(self.unresampled)
        res=un[0]
        pup=un[5]
        idx=self.plot_stimidx
        b=np.nanmean(pup[:,:,idx],axis=0)
        bc=np.asarray(sorted(zip(b,range(0,len(b)))),dtype=int)
        bc=bc[:,1]
        res[:,:,idx]=res[:,bc,idx]
        un[0]=res
        nu.raster_plot(data=un,stims=idx,size=(12,6))
           

            
# end nems_stack


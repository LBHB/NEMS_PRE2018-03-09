import numpy as np
import scipy.io
import scipy.signal
import copy

empty_data={}

class nems_data:
    """nems_data

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
    input_name='stim'  # name of input matrix in d_in
    output_name='stim' # name of output matrix in d_out
    phi=None # vector of parameter values that can be fit
    d_in=None
    d_out=None
    name='pass-through'
    
    def __init__(self,d_in=None):
        self.data_setup(d_in)
        
    def data_setup(self,d_in=None):
        if d_in is None:
            self.d_in=nems_data() # list of data buckets fed into module
        else:
            self.d_in=d_in
        self.d_out=nems_data() # list of outputs, same size as data in

    def prep_eval(self):
        self.d_out.copy_keys(self.d_in)
        
    def eval(self):
        # default: pass-through pointers to data from input to output,
        # need to deepcopy individual dict entries if they are changed
        self.prep_eval()
        
    def auto_plot(self):
        print("dummy auto_plot")
        
# end nems_module

class load_mat(nems_module):

    name='load_mat'
     
    def __init__(self,d_in=None,est_files=[],val_files=[]):
        self.data_setup(d_in)
        self.d_in.est_files=est_files
        self.d_in.vall_files=val_files

    def eval(self):
        self.prep_eval()

        # new list object for dat
        self.d_out.data=copy.deepcopy(self.d_in.data)
        
        # load contents of Matlab data file
        for f in self.d_in.est_files:
            #f='tor_data_por073b-b1.mat'
            print(f)
            data = scipy.io.loadmat(f,chars_as_strings=True)
            
            # append contents of file to data, assuming data is a dictionary
            # with entries stim, resp, etc...
            self.d_out.data.append(data)

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
        
class add_scalar(nems_module):
 
    name='add_scalar'
    
    def __init__(self, d_in=None, n=1):
        self.data_setup()
        self.n=n
       
    def eval(self):
        self.prep_eval()
        self.d_out.data=copy.deepcopy(self.d_in.data)

        for f in self.d_out.data:
            f[self.output_name]=f[self.input_name]+self.n
        
class sum_dim(nems_module):
    name='sum_dim'
     
    def __init__(self, d_in=None, dim=1):
        self.data_setup(d_in)
        self.dim=dim
        
    def eval(self):
        self.prep_eval()

        self.d_out.data=copy.deepcopy(self.d_in.data)

        for f in self.d_out.data:
            f[self.output_name]=f[self.input_name].sum(axis=self.dim)

class nems_stack:
    """nems_stack

    Properties:
     modules = list of nems_modules in sequence of execution

    """
     
    def __init__(self):
        print("dummy")
        self.modules=[]

    def eval(self,start=0):
        # evalute stack, starting at module # start
        for ii in range(start,len(self.modules)):
            #if ii>0:
            #    print("Propagating mod {0} d_out to mod{1} d_in".format(ii-1,ii))
            #    self.modules[ii].d_in=self.modules[ii-1].d_out
            self.modules[ii].eval()
            
    def append(self, mod=None):
        if mod is None:
            mod=nems_module()
        if len(self.modules):
            print("Propagating d_out from {0} into new d_in".format(self.modules[-1].name))
            mod.d_in=self.modules[-1].d_out
        self.modules.append(mod)
        
    def popmodule(self, mod=nems_module()):
        del self.modules[-1]
        
    def output(self):
        if len(self.modules)==0:
            return {}
        else:
            return self.modules[-1].d_out
        
        
# end nems_stack

est_files=['/Users/svd/python/nems/ref/week5_TORCs/tor_data_por073b-b1.mat']

stack=nems_stack()
stack.append(load_mat(est_files=est_files))
stack.append()

stack.eval()
out=stack.output()
print('stim[0][0]: {0}'.format(out.d(0)['stim'][0][0]))

stack.append(add_scalar(n=2))
stack.eval(1)
out=stack.output()
print('stim[0][0]: {0}'.format(out.d(0)['stim'][0][0]))

stack.append(sum_dim(dim=2))
stack.eval(1)
out=stack.output()
print('stim[0][0]: {0}'.format(out.d(0)['stim'][0][0]))


#print('Pre eval')
#stack.output()

#stack.eval()

#print('Output post eval')
#stack.output()
 
        

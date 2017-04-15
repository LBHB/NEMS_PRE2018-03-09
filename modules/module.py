import numpy as np


class nems_data:
    """nems_data

    Generic NEMS data bucket

    provides input and output of each nems_module

    structure containing a set of matrices, corresponding to input(s)
    and output(s). eg, resp, stim, stim2, state, etc.

    """

    stim=None
    resp=None
    state=None
    
    def __init__(self, name='blank'):
        self.name=name  

# end nems_data

        
        
class nems_module:
    """nems_module

    Generic NEMS module

    """
    
    data_in=[]  # list of data buckets fed into module
    data_out=[]   # list of outputs, same size as data in
    
    input_name='stim'  # name of matrix in data_in that should
                       # provide input to eval
    output_name='stim'
    
    phi=None #vector of parameter values that can be fit
    
    def __init__(self):
        self.data_in=[]
        
    def eval(self):
        # default: pass-through data from input to output
        del self.data_out[:]
        self.data_out.extend(self.data_in)
        
    def auto_plot(self):
        print("dummy")
        
# end nems_module

        
        
class nems_stack:
    """nems_stack

    Array?/Dictionary? of nems_modules in sequence of execution

    """
    modules=[]
    
    def __init__(self):
        print("dummy")
        
    def eval(self,start=0):
        # evalute stack, starting at module # start
        for ii in range(start,len(self.modules)):
            self.modules[ii].eval()
            
    def addmodule(self, mod=nems_module()):
        if len(self.modules):
            mod.data_in=self.modules[-1].data_out
        self.modules.append(mod)
        
    def popmodule(self, mod=nems_module()):
        del self.modules[-1]
        
    def output(self):
        return self.modules[-1].data_out
        
        
 # end nems_stack
       
 stack=nems_stack()
 stack.addmodule()
 stack.addmodule()
 stack.modules[0].data_in.extend([1,3,2])
 
 stack.output()
 
 stack.eval()
 
 stack.output()
 
        

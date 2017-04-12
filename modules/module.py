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

        

class nems_data_set:
    """nems_data_set

    Generic NEMS data bucket

    array (dictionary) of nems_data objects that are input and output
    from each module

    """
    
    def __init__(self):

# end nems_data_set

       
        
class nems_module:
    """nems_module

    Generic NEMS module

    """
    
    data_in=None  # nems_data_set before transformation by module
                  # should point to output of previous module in stack
    data_out=None   # nems_data_set after transformation by module
    
    input_name='stim'  # name of matrix in data_in that should
                       # provide input to eval
    output_name='stim'
    
    phi=None #vector of parameter values that can be fit
    
    def __init__(self):
        
        
    def eval(self):
        # default pass-through data from input to output
        self.data_out=self.data_in
        
    def auto_plot(self):
        
        
# end nems_module

        
        
class nems_stack:
    """nems_stack

    Array?/Dictionary? of nems_modules in sequence of execution

    """
    modules=None
    
    def __init__(self):

    def eval(start=0):
        # evalute stack, starting at module # start
        for ii in range(start,count(self.modules)):
            self.modules.eval()
            
 # end nems_stack
       
        

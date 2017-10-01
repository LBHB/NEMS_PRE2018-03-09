#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 12:26:54 2017

@author: shofer
"""
#Needs to be imported for all nems_module child classes:
from nems.modules.base import nems_module 
import nems.utilities as ut

#Import specific to this module:
import numpy as np
import copy 

class simple_demo(nems_module):
    """
    Applies a simple DC gain and offset to the input data with a 0 threshold:
        y = v1*x + v2
    where x is the input variable, and v1,v2 are fitted parameters. 
    If y<0, it sets y=0. 
    
    This module is designed to show how to set up a custom 
    nems_module child class. For most nems_module child classes, there are two
    important class methods: my_init and my_eval. my_init initializes parameters
    specific to the child class that aren't created by the base nems_module 
    parent class, while my_eval takes input data, applies some transformation,
    and returns an output.
    
    Some classes that don't just have a simple input and output require different
    method definitions, and are covered in the advanced_demo module.
    
    Once the module is defined, look at nems.keywords to see what goes into a 
    keyword. Keywords are how modules are applied to the nems_stack object. 
    
    NOTE: this module should be imported into the user_def package __init__ file.
    """
    name='user_def.demo.simple_demo' #name of module, starting from modules package
    plot_fns=[ut.plot.pre_post_psth] #plot function to be used for this module
    
    
    def my_init(self,fit_fields=['phi'],phi=[1,0],thresh=False):
        """
        field_dict: dictionary of inputs (including default inputs) passed to
                    function. Needed for saving & reloading a fitted stack 
                    without pickling the entire object, as it is much more space
                    efficient to save dictionaries as a JSON file than entire 
                    objects as pickles.
        fit_fields: name of fitted parameters
        phi: fitted parameters. The values input serve as the inital value for 
                fitting. 
        thresh: Boolean to turn on or off output threshold
        
        """
        self.field_dict=locals() #create dictionary of inputs passed to function
        self.field_dict.pop('self',None) #remove self from input dict
        
        self.fit_fields=fit_fields 
        self.phi=np.array(phi) 
        self.thresh=thresh 
        
    def my_eval(self,X):
        """
        my_eval should always take self and X as its inputs, as these are what 
        are passed by the parent class nems_module. X is the input data, which 
        is usually the 'stim' key of stack.data. It should output an array Y, 
        which is usually the same shape as the input (though not necessarily, 
        see fir or weight_channel modules)
        """
        v1=self.phi[0]
        v2=self.phi[1]
        Y=v1*X + v2
        if self.thresh is True:
            indices=Y<0.0
            Y[indices]=0
        return Y
    
    
class adv_demo(nems_module):
    """
    Applies a simple DC pupil gain to the input data:
        y=v1*x + v2*p + v3
    where x is the input data, p is the pupil data, and v1,v2,v3 are fitted 
    parameters.
    
    This module is designed to show how to set up a more advaced nems_module 
    child class. Where the simple_demo module uses a my_eval() function that just 
    takes a simple input array and outputs another array, advanced_demo will 
    use an evaluate() function that will modify parameters of the nems_stack it 
    is associated to. 
    """
    name='user_def.demo.adv_demo'
    plot_fns=[ut.plot.pre_post_psth]
    
    
    def my_init(self,fit_fields=['theta'],theta=[1,0,0]):
        self.field_dict=locals() 
        self.field_dict.pop('self',None)
        self.fit_fields=fit_fields
        self.theta=np.array(theta)
        
    def evaluate(self,nest=0):
        """
        Copies over the input data dictionary (or list of dictionaries) to the 
        next element of stack.data, then replaces the relevant entries in the 
        dictionary with the new, transformed values. 
        
        evaluate should be used instead of my_eval when the transformation 
        performed by the function is not a simple in/out function, with one 
        input and one output value.
        """
        if nest==0:
            """
            If this is the first time the function is being evaluated, 
            copy over the input data dictionary to the next entry in the 
            nems_stack
            """
            del self.d_out[:]
            for i,val in enumerate(self.d_in):
                self.d_out.append(copy.deepcopy(val))
        for f_in,f_out in zip(self.d_in,self.d_out):
            #For each input dictionary in the current level of the nems_stack:
            if self.parent_stack.nests>0 and f_in['est'] is False:
                """
                If 'est' is False, it indicates that this dictionary is validation
                data. Since validation data is often nested, we need to specify 
                which nest we are currently evaluating
                """
                X=copy.deepcopy(f_in[self.input_name][nest])
                Xp=copy.deepcopy(f_in[self.state_var][nest])
                Y=self.theta[0]*X + self.theta[1]*Xp + self.theta[2]
                f_out[self.output_name][nest]=Y
            else:
                """
                If 'est' is True, this is estimation data, which means we are 
                fitting, and thus don't need to worry about the nests.
                """
                X=copy.deepcopy(f_in[self.input_name])
                Xp=copy.deepcopy(f_in[self.state_var])
                Y=self.theta[0]*X + self.theta[1]*Xp + self.theta[2]
                f_out[self.output_name]=Y

    
    
    
    
    
    
    
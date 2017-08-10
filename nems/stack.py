#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:54:26 2017

@author: shofer
"""

import matplotlib.pyplot as plt, mpld3
import nems.utilities as ut
import numpy as np
import os
import io

try:
    import boto3
    import nems_config.Storage_Config as sc
    AWS = sc.USE_AWS
except:
    from nems_config.defaults import STORAGE_DEFAULTS
    sc = STORAGE_DEFAULTS
    AWS = False

class nems_stack:
        
    """
    Key components:
        modules: list of nems_modules in sequence of execution
        data: stream of data as it is evaluated through the sequence of modules
        fitter: pointer to the fit module
        quick_plot: generates a plot of something about the transformation
                     that takes place at each modules step
        nests: number validation sets to create. Used for nested crossvalidation. 
                If 5% validation sets are wanted, use 20 nests, etc.
        parm_fits: The fitted parameters for each nest. A list of phi vectors.
        plot_dataidx: which dataset to plot (estimation or validation). Usually, 
                        0 is the estimation set and 1 is the validation set, but
                        sometimes there are more sets. Even numbers (and zero) 
                        are estimation sets, while odd numbers are validation sets.
        plot_stimidx: which stimulus to plot
        
    

    """
    
    #modelname=None
    modules=[]  # stack of modules
    mod_names=[]
    mod_ids=[]
    data=[]     # corresponding stack of data in/out for each module
    meta={}
    fitter=None
    valmode=False
    nests=1
    avg_resp=True
    plot_dataidx=0
    plot_stimidx=0
    parm_fits=[]
    fitted_modules=[]
    cv_counter=0
    keywords=[]
    valfrac=0.05
    parm_fits=[]
    
    def __init__(self):
        print("Creating new stack")
        self.modules=[]
        self.mod_names=[]
        self.data=[]
        self.data.append([])
        self.data[0].append({})    #Also this?
        self.data[0][0]['resp']=[] #Do we need these? 
        self.data[0][0]['stim']=[] #This one too?
        
        self.meta={}  #Dictionary that will contain cellid, batch, modelname
        #self.modelname='Empty stack'
        self.error=self.default_error
        self.valmode=False
        self.unresampled=[] #If the data is resampled by load_mat, holds an unresampled copy for raster plot
        self.nests=1 #Default is to have only one nest, i.e. standard crossval
        self.parm_fits=[] #List of fitted parameters for each nest
        self.fitted_modules=[] #List of modules with fitted parameters
        self.cv_counter=0 #Counter for iterating through nests, used in nm.crossval
        self.keywords=[] #The split modelname string
        self.mod_ids=[]
        self.valfrac=0.05 #Fraction of the data used to create each validation nest
        
    def evaluate(self,start=0):
        """
        Evaluate modules in stack, starting at module # start. When stack is 
        fitting (valmode is False), simply calls mod.evaluate() for each module
        in the stack. However, when the stack is testing (valmode is True), 
        evaluate first calls an est/val function to create a list of validation
        datasets ("nest"). It then extracts the fitted parameters for each list, and calls
        mod.evaluate for each module in the stack for each validation set in the
        list. 
        
        Note that during valmode, the estimation dataset that is returned is the
        last dataset fit.
        """
        if self.valmode is True: 
            print('Evaluating validation data')
            mse_idx=ut.utils.find_modules(self,'metrics.mean_square_error')
            mse_idx=int(mse_idx[0])
            try:
                xval_idx=ut.utils.find_modules(self,'est_val.crossval')
            except:
                xval_idx=ut.utils.find_modules(self,'est_val.standard')
            xval_idx=xval_idx[0]
            if start !=0 and start<=xval_idx:
                self.modules[xval_idx].evaluate()
                start=xval_idx+1
            for ii in range(start,mse_idx):
                for i in range(0,self.nests):
                    st=0
                    for m in self.fitted_modules:
                        phi_old=self.modules[m].parms2phi()
                        s=phi_old.shape
                        self.modules[m].phi2parms(self.parm_fits[i][st:(st+np.prod(s))])
                        st+=np.prod(s)
                    self.modules[ii].evaluate(nest=i)
            ut.utils.concatenate_helper(self,start=xval_idx+1,end=mse_idx+1)
            for ij in range(mse_idx,len(self.modules)):
                self.modules[ij].evaluate() 
        else:
            #This condition evaluates for fitting
            for ii in range(start,len(self.modules)):
                self.modules[ii].evaluate() 
            
    
    # create instance of mod and append to stack    
    def append(self, mod=None, **xargs):
        """
        Creates an instance of a module and appends it to the stack. Evaluates 
        module in doing so. 
        """
        if mod is None:
            raise ValueError('stack.append: module not specified')
        else:
            m=mod(self, **xargs)
        self.append_instance(m)
        
    def append_instance(self, mod=None):
        """Same as append but takes an instance of a module instead
        of the class to preserve existing **xargs. For use with insert/remove.
        Could maybe merge these with an added boolean arg? Wasn't sure if that
        would interfere with the **xargs.
        
        @author: jacob
        
        """
        if not mod:
            raise ValueError('stack.append: module not specified')
            #mod=nm.nems_module(self)
        self.modules.append(mod)
        self.data.append(mod.d_out)
        self.mod_names.append(mod.name)
        self.mod_ids.append(mod.idm)
        
        mod.evaluate()
        
    def append_no_eval(self,mod=None,**xargs):
        """
        Creates an instance of a module and appends it to the stack. Evaluates 
        module in doing so. 
        """
        if mod is None:
            raise ValueError('stack.append: module not specified')
        else:
            m=mod(self, **xargs)
        self.modules.append(mod)
        self.data.append(mod.d_out)
        self.mod_names.append(mod.name)
        self.mod_ids.append(mod.idm)
        
        
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
            given, ut.utils.find_modules will be used to find idx.
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
            idx = ut.utils.find_modules(self, mod.name)
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
        
    def popmodule(self):
        del self.modules[-1]
        del self.data[-1]
        
    def clear(self):
        del self.modules[:]
        del self.data[1:]
        
    def output(self):
        return self.data[-1]
    
    def default_error(self):
        return np.zeros([1,1])
    
    def quick_plot(self,size=(12,24)):
        plt.figure(figsize=size)
        
        # find all modules with plotfn
        plot_set=[]
        for idx,m in enumerate(self.modules):
            if m.auto_plot:
                plot_set.append(idx)
        
        spidx=1
        for idx in plot_set:
            print("quick_plot: {}".format(self.modules[idx].name))
            plt.subplot(len(plot_set),1,spidx)
            #plt.subplot(len(self.modules),1,idx+1)
            self.modules[idx].do_plot(self.modules[idx])
            spidx+=1
        
        #plt.tight_layout()
        #TODO: Use gridspec to fix spacing issue? Addition of labels makes
        #      the subplots look "scrunched" vertically.
    
    def quick_plot_save(self, mode='png'):
        """Copy of quick_plot for easy save or embed.
        
        mode options:
        -------------
        "json" -- .json
        "html" -- .html
        "png" -- .png
        default -- .png
        
        returns:
        --------
        filename : string
            Path to saved file, currently of the form:
            "/auto/data/code/nems_saved_images/batch{#}/{cell}/{modelname}.type"
                        
        @author: jacob
        
        """
        batch = self.meta['batch']
        cellid = self.meta['cellid']
        modelname = self.meta['modelname']
    
        fig = plt.figure(figsize=(8,9))
        for idx,m in enumerate(self.modules):
        # skip first module
            if idx>0:
                print(self.mod_names[idx])
                plt.subplot(len(self.modules)-1,1,idx)
                m.do_plot(m)
        plt.tight_layout()

        filename = (
                    sc.DIRECTORY_ROOT + "nems_saved_images/batch{0}/{1}/{2}.{3}"
                    .format(batch, cellid, modelname, mode)
                    )
        
        if AWS:
            s3 = boto3.resource('s3')
            key = filename[len(sc.DIRECTORY_ROOT):]
            fileobj = io.BytesIO()
            fig.savefig(fileobj, format=mode)
            fileobj.seek(0)
            s3.Object(sc.PRIMARY_BUCKET, key).put(Body=fileobj)
        else:
            if os.path.isfile(filename):
                os.remove(filename)
            try:
                fig.savefig(filename)
            except Exception as e:
                print("Bad file extension for figure or couldn't save")
                print(e)
                raise e
        
            try:
                os.chmod(filename, 0o666)
            except Exception as e:
                print("Couldn't modify file permissions for figure")
                raise e

        return filename
    


                




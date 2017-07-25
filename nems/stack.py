#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:54:26 2017

@author: shofer
"""

import nems.modules as nm
import matplotlib.pyplot as plt,mpld3
import nems.utils as nu
import numpy as np
import copy

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
    #TODO: maybe in the future we want to put nems_stack into its own file? Would 
    #probably take some work, but might reduce clutter in this file ---njs 13 July 2017
    
    modelname=None
    modules=[]  # stack of modules
    mod_names=[]
    mod_ids=[]
    data=[]     # corresponding stack of data in/out for each module
    meta={}
    fitter=None
    valmode=False
    #cross_val=False
    #nests=1
    avg_resp=True
    
    plot_dataidx=0
    plot_stimidx=0
    fitted_modules=[]
    cv_counter=0
    
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
        self.unresampled=[] #If the data is resampled by load_mat, holds an unresampled copy for raster plot
        self.nests=1 #Default is to have only one nest, i.e. standard crossval
        self.parm_fits=[]
        self.fitted_modules=False
        self.cv_counter=0
        
    def evaluate(self,start=0):
        """
        evaluate stack, starting at module # start
        """
        if self.valmode is True: #and self.pre_flag is not True:
            #xval_idx=nu.find_modules(self,'crossval')
            #xval_idx=xval_idx[0]
            #print(xval_idx)
            #self.modules[xval_idx].evaluate()
            #print(self.data[2][1]['stim'][3].shape)
            
            #TODO: there is definitely a more efficient way to do this, but this
            #works reliably. In the future, should make it so that crossval is
            #only evaluated once ---njs 24 July 2017
            
            for i in range(0,self.nests):
                for j in range(start,len(self.modules)):
                    st=0
                    for m in self.fitted_modules:
                        phi_old=self.modules[m].parms2phi()
                        s=phi_old.shape
                        self.modules[m].phi2parms(self.parm_fits[i][st:(st+np.prod(s))])
                        st+=np.prod(s)
                    self.modules[j].evaluate(nest=i)
            for k in range(start+1,len(self.data)):
                for n in range(0,len(self.data[-1])):
                    if n%2 !=0:
                        if self.data[k][n]['stim'][0].ndim==3:
                            self.data[k][n]['stim']=np.concatenate(self.data[k][n]['stim'],axis=1)
                        else:
                            self.data[k][n]['stim']=np.concatenate(self.data[k][n]['stim'],axis=0)
                        self.data[k][n]['resp']=np.concatenate(self.data[k][n]['resp'],axis=0)
                        try:
                            self.data[k][n]['pupil']=np.concatenate(self.data[k][n]['pupil'],axis=0)
                        except ValueError:
                            print('No pupil data')
                            self.data[k][n]['pupil']=None
                        try:
                            self.data[k][n]['replist']=np.concatenate(self.data[k][n]['replist'],axis=0)
                        except ValueError:
                            print('Trials average, used repcount')
                            self.data[k][n]['replist']=[]
                            
        else:
            for ii in range(start,len(self.modules)):
                self.modules[ii].evaluate() 
            
    
    # create instance of mod and append to stack    
    def append(self, mod=None, **xargs):
        if mod is None:
            m=nm.nems_module(self)
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
            mod=nm.nems_module(self)
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
        
    def popmodule(self, mod=nm.nems_module()):
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
        plt.subplot(len(self.modules)-1,1,1)
        #self.do_raster_plot()
        for idx,m in enumerate(self.modules):
            # skip first module
            if idx>0:
                print(self.mod_names[idx])
                plt.subplot(len(self.modules)-1,1,idx)
                #plt.subplot(len(self.modules),1,idx+1)
                m.do_plot(m)
        #plt.tight_layout()
        #TODO: Use gridspec to fix spacing issue? Addition of labels makes
        #      the subplots look "scrunched" vertically.
    
    def quick_plot_save(self, mode=None):
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
            "/auto/data/code/nems_saved_models/batch{#}/{cell}_{modelname}.type"
                        
        @author: jacob
        
        """
        batch = self.meta['batch']
        cellid = self.meta['cellid']
        modelname = self.meta['modelname']
    
        fig = plt.figure(figsize=(8,9))
        for idx,m in enumerate(self.modules):
        # skip first module
            if idx>0:
                plt.subplot(len(self.modules)-1,1,idx)
                m.do_plot(m)
        plt.tight_layout()
        
        file_root = (
                "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.",
                [batch, cellid, modelname]
                )
        if mode is not None:
            mode = mode.lower()
        if mode is None:
            #filename = (
            #        "/auto/data/code/nems_saved_models/batch{0}/{1}_{2}.png"
            #        .format(batch,cellid,modelname)
            #        )
            filename = (file_root[0] + 'png').format(*file_root[1])
            fig.savefig(filename)
        elif mode == "png":
            filename = (file_root[0] + 'png').format(*file_root[1])
            fig.savefig(filename)
        elif mode == "pdf":
            filename = (file_root[0] + 'pdf').format(*file_root[1])
            fig.savefig(format="pdf")
        elif mode == "svg":
            filename = (file_root[0] + 'svg').format(*file_root[1])
            fig.savefig(format="svg")
        elif mode == "json":
            filename = (file_root[0] + 'JSON').format(*file_root[1])
            mpld3.save_json(fig, filename)
        elif mode == "html":
            filename = (file_root[0] + 'html').format(*file_root[1])
            mpld3.save_html(fig, filename)
        else:
            print("%s is not a valid format -- saving as .png instead."%mode)
            filename = (file_root[0] + 'png').format(*file_root[1])
            fig.savefig(filename)
        plt.close(fig)
        return filename
    
#    def trial_quick_plot(self):
#        """
#        Plots several trials of a stimulus after fitting pupil data.
#        This is to make it easier to visualize the fits on individual trials,
#        as opposed to over the entire length of the fitted vector.
#        """
#        plt.figure(figsize=(12,15))
#        for idx,m in enumerate(self.modules):
#            # skip first module
#            if idx>0:
#                plt.subplot(len(self.modules)-1,1,idx)
#                m.do_trial_plot(m,idx)

                




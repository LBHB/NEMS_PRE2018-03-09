#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 13:21:47 2017

@author: shofer
"""
from nems.modules.base import nems_module
import numpy as np
import copy
import math as mt

import nems.utilities.utils
import nems.utilities.plot


class standard(nems_module):
    """
    Splits stack.data object into estimation and validation datasets. If 
    estimation and validation datasets are already flagged in stack.data (i.e.
    if d['est'] exists), it will simply pass the datasets as is. If not, it will
    split the data based on the 'repcount' list. Stimuli with large numbers of
    repetitions are placed in the validation dataset, while most of the stimuli,
    which have low repetitions, are placed in the estimation dataset.
    
    This estimation/validation routine is not compatible with nested
    crossvalidation.
    """
    name='est_val.standard'
    user_editable_fields=['input_name','output_name','valfrac']
    valfrac=0.05
    def my_init(self,valfrac=0.05):
        self.field_dict=locals()
        self.field_dict.pop('self',None)
        self.valfrac=valfrac
        print('Using standard est/val')
    
    def evaluate(self,**kwargs):
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
                s=copy.deepcopy(d['repcount'])
                
                m=s.max()
                validx = s==m
                estidx = s<m
                if not estidx.sum():
                    s[-1]+=1
                    m=s.max()
                    validx = s==m
                    estidx = s<m
                
                d_est=d.copy()
                #d_val=d.copy()
                
                d_est['repcount']=copy.deepcopy(d['repcount'][estidx])
                d_est['resp']=copy.deepcopy(d['resp'][estidx,:])
                d_est['stim']=copy.deepcopy(d['stim'][:,estidx,:])
                #d_val['repcount']=copy.deepcopy(d['repcount'][validx])
                #d_val['resp']=copy.deepcopy(d['resp'][validx,:])
                #d_val['stim']=copy.deepcopy(d['stim'][:,validx,:])
                try:
                    if d['pupil'].size:
                        d_est['pupil']=copy.deepcopy(d['pupil'][estidx,:])
                except:
                    pass
                    #print('No pupil data')
                    
                d_est['est']=True
                #d_val['est']=False
                
                self.d_out.append(d_est)
                if self.parent_stack.valmode:
                    
                    d_val=d.copy()
                    d_val['repcount']=copy.deepcopy(d['repcount'][validx])
                    d_val['resp']=copy.deepcopy(d['resp'][validx,:])
                    d_val['stim']=copy.deepcopy(d['stim'][:,validx,:])
                    try:
                        if d['pupil'].size:
                            d_val['pupil']=copy.deepcopy(d['pupil'][validx,:])
                    except:
                        pass
                        #print('No pupil data')
                        
                    d_val['est']=False
                    self.d_out.append(d_val)

            
class crossval2(nems_module):
    """
    Splits data into estimation and validation datasets. If estimation and 
    validation sets are already flagged (if d['est'] exists), it just passes 
    these. If not, it splits a given percentage of the dataset off as validation
    data, and leaves the rest as estimation data. 
    
    Inputs:
        valfrac: fraction of the dataset to allocate as validation data
    
    This module is set up to work with nested crossvalidation. If this is the 
    case, it will run through the dataset, taking a different validation set 
    each time. 
    
    @author: shofer
    """
    name='est_val.crossval2'
    user_editable_fields=['input_name','output_name',
                          'valfrac','interleave_valtrials','val_mult_repeats',
                          'cv_counter']
    plot_fns=[nems.utilities.plot.raster_plot,nems.utilities.plot.plot_spectrogram]
    valfrac=0.05
    interleave_valtrials=True
    val_mult_repeats=True
    cv_counter=0
    nests=0
    validx_sets=[]
    
    def my_init(self,valfrac=0,interleave_valtrials=True,val_mult_repeats=True):
        #self.field_dict=locals()
        #self.field_dict.pop('self',None)
        nests=self.parent_stack.meta['nests']
        #if nests==0:
        #    nests=1;
        #    self.parent_stack.nests=1
        if nests>1:
            valfrac=1/nests
        elif valfrac==0:
            valfrac=0.05
            
        self.valfrac=valfrac
        self.validx_sets=[]
        self.nests=nests
        self.interleave_valtrials=interleave_valtrials
        self.val_mult_repeats=val_mult_repeats
        self.cv_counter=0
        
    def evaluate(self,nest=0):

        del self.d_out[:]

        for i, d in enumerate(self.d_in):
            valfrac=self.valfrac
            try:
                count=self.parent_stack.meta['cv_counter']
            except:
                count=self.cv_counter
                
            nests=int(1/valfrac)
            
            re=d['resp'].shape
            spl=re[0]*valfrac
            if re[0]<nests:
                raise IndexError('Fewer stimuli than nests; use a higher valfrac/less nests')

            # figure out grouping for each CV set 
            if self.interleave_valtrials:
                smax=np.int(np.ceil(spl))
                a=np.arange(np.ceil(spl)*nests).astype(int)
                a=np.reshape(a,[smax,nests])
                a=a.transpose()
                a=np.reshape(a,[smax*nests])
                a=a[a<re[0]]
            else:
                a=np.arange(re[0]).astype(int)
            
            self.validx_sets=[]
            for cc in range(0,nests):
                c1=mt.floor((cc)*spl)
                c2=mt.floor((cc+1)*spl)
                self.validx_sets.append(a[c1:c2])
                
            c1=mt.floor((count)*spl)
            c2=mt.floor((count+1)*spl)
            print("Nest {0}/{1}, File {2}, c1-c2: {3}-{4}".format(count,nests,i,c1,c2))
            nidx=self.validx_sets[count]
            #print(nidx)
            
            d_est=d.copy()
            d_val=d.copy()               

            d_est['est']=True
            d_val['est']=False
            
            d_est['resp']=np.delete(d['resp'],np.s_[nidx],0)
            d_val['resp']=copy.deepcopy(d['resp'][nidx,:])
            
            d_est['stim']=np.delete(d['stim'],np.s_[nidx],1)
            d_val['stim']=copy.deepcopy(d['stim'][:,nidx,:])

            try:
                d_est['pupil']=np.delete(d['pupil'],np.s_[nidx],0)
                d_val['pupil']=copy.deepcopy(d['pupil'][nidx,:])
            except:
                #print('No pupil data')
                d_est['pupil']=[]
                d_val['pupil']=[]
            
            try:
                d_est['repcount']=np.delete(d['repcount'],np.s_[nidx],0)
                d_val['repcount']=copy.deepcopy(d['repcount'][nidx])
            except:
                d_est['repcount']=None
                d_val['repcount']=None
                
            try:
                d_est['replist']=np.delete(d['replist'],np.s_[nidx],0)
                d_val['replist']=copy.deepcopy(d['replist'][nidx])
            except:
                d_est['replist']=None
                d_val['replist']=None
                
            self.d_out.append(d_est)
            if self.parent_stack.valmode is True:
                self.d_out.append(d_val)
                        
            #if self.cv_counter==self.nests-1:
            #    self.parent_stack.cond=True
                    

class crossval(nems_module):
    """
    Splits data into estimation and validation datasets. If estimation and 
    validation sets are already flagged (if d['est'] exists), it just passes 
    these. If not, it splits a given percentage of the dataset off as validation
    data, and leaves the rest as estimation data. 
    
    Inputs:
        valfrac: fraction of the dataset to allocate as validation data
    
    This module is set up to work with nested crossvalidation. If this is the 
    case, it will run through the dataset, taking a different validation set 
    each time. 
    
    @author: shofer
    """
    name='est_val.crossval'
    user_editable_fields=['input_name','output_name',
                          'valfrac','interleave_valtrials','val_mult_repeats',
                          'cv_counter']
    plot_fns=[nems.utilities.plot.raster_plot,nems.utilities.plot.plot_spectrogram]
    valfrac=0.05
    interleave_valtrials=True
    val_mult_repeats=True
    cv_counter=0
    nests=0
    
    def my_init(self,valfrac=0,interleave_valtrials=True,val_mult_repeats=True):
        #self.field_dict=locals()
        #self.field_dict.pop('self',None)
        nests=self.parent_stack.nests
        if nests==0:
            nests=1;
            self.parent_stack.nests=1
        if nests>1:
            valfrac=1/nests
        elif valfrac==0:
            valfrac=0.05
            
        self.valfrac=valfrac
        self.nests=nests
        self.interleave_valtrials=interleave_valtrials
        self.val_mult_repeats=val_mult_repeats
        self.cv_counter=0
        
    def evaluate(self,nest=0):

        del self.d_out[:]

        for i, d in enumerate(self.d_in):
            try:
                if d['est']:
                    # flagged as est data
                    self.d_out.append(d)
                elif self.parent_stack.valmode:
                    self.d_out.append(d)
                self.parent_stack.cond=True
                self.parent_stack.pre_flag=True
            except:
                valfrac=self.valfrac
                count=self.cv_counter
                nests=int(1/valfrac)
                avg_resp=self.parent_stack.avg_resp
                
                re=d['resp'].shape
                if re[0]<self.parent_stack.nests:
                    raise IndexError('Fewer stimuli than nests; use a higher valfrac/less nests')
                
                #a=np.find(np.isfinite(d['resp'][:,0]))
                
                spl=re[0]*valfrac
                if self.interleave_valtrials:
                    smax=np.int(np.ceil(spl))
                    a=np.arange(np.ceil(spl)*nests).astype(int)
                    a=np.reshape(a,[smax,nests])
                    a=a.transpose()
                    a=np.reshape(a,[smax*nests])
                    a=a[a<re[0]]
                else:
                    a=np.arange(re[0]).astype(int)
                
                c1=mt.floor((count)*spl)
                c2=mt.floor((count+1)*spl)
                print("Nest {0}, File {1}, c1-c2: {2}-{3}".format(count,i,c1,c2))
                nidx=a[c1:c2]
                print(nidx)
                d_est=d.copy()
                
                d_est['resp']=np.delete(d['resp'],np.s_[nidx],0)
                d_est['stim']=np.delete(d['stim'],np.s_[nidx],1)
                
                if self.parent_stack.avg_resp is True:
                    try:
                        d_est['pupil']=np.delete(d['pupil'],np.s_[nidx],2)
                    except TypeError:
                        print('No pupil data')
                        d_est['pupil']=[]  
                    d_est['repcount']=np.delete(d['repcount'],np.s_[nidx],0)
                else:
                    try:
                        d_est['pupil']=np.delete(d['pupil'],np.s_[nidx],0)
                    except TypeError:
                        print('No pupil data')
                        d_est['pupil']=[]
                    try:
                        d_est['replist']=np.delete(d['replist'],np.s_[nidx],0)
                    except KeyError:
                        d_est['replist']=None
                

                d_est['est']=True
                
                self.d_out.append(d_est)
                if self.parent_stack.valmode is True:
                    
                    d_val=d.copy()
                    d_val['est']=False
                    
                    d_val['stim']=[]
                    d_val['resp']=[]
                    d_val['pupil']=[]
                    d_val['replist']=[]
                    d_val['repcount']=[]

                    import copy
                    for count in range(0,self.parent_stack.nests):
                        #print(count)
                        re=d['resp'].shape
                        spl=re[0]*valfrac
                        c1=mt.floor((count)*spl)
                        c2=mt.floor((count+1)*spl)
                        nidx=a[c1:c2]
                        print("V nest {0}, File {1}, c1-c2: {2}-{3}".format(count,i,c1,c2))
                        print(nidx)
                        if avg_resp:
                            try:
                                d_val['pupil'].append(copy.deepcopy(d['pupil'][:,:,nidx]))
                            except TypeError:
                                print('No pupil data')
                                d_val['pupil']=[]
                            d_val['repcount'].append(copy.deepcopy(d['repcount'][nidx]))
                        else:
                            try:
                                d_val['pupil'].append(copy.deepcopy(d['pupil'][nidx,:]))
                            except TypeError:
                                print('No pupil data')
                                d_val['pupil']=[]
                            d_val['replist'].append(copy.deepcopy(d['replist'][nidx]))
                            d_val['repcount']=copy.deepcopy(d['repcount'])
                        d_val['resp'].append(copy.deepcopy(d['resp'][nidx,:]))
                        d_val['stim'].append(copy.deepcopy(d['stim'][:,nidx,:]))
                        
                      
                    #TODO: this code runs if crossval allocated
                    #an empty nest at the end of the validation list. This 
                    #should not happen as often as it does, and it would be a 
                    #better long term thing to do to change how the indices for
                    #allocating the datasets are chosen (something better than 
                    #mt.ceil), since then estimation nests with no validation
                    #nest would not be fit, as they are currently.
                    #    ----njs, August 2 2017
                    """
                    s=d_val['stim'][-1].shape
                    sr=d_val['resp'][-1].shape
                    while s[1]==0 or sr[0]==0:
                        del(d_val['stim'][-1])
                        del(d_val['resp'][-1])
                        del(d_val['pupil'][-1])
                        try:
                            del(d_val['replist'][-1])
                        except:
                            pass
                        self.parent_stack.nests-=1
                        s=d_val['stim'][-1].shape
                        sr=d_val['resp'][-1].shape
                        print('Final nest has no stimuli, updating to have {0} nests'.format(
                                self.parent_stack.nests))
                    """
                    self.d_out.append(d_val)
                
                if self.cv_counter==self.nests-1:
                    self.parent_stack.cond=True
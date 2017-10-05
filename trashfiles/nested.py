#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nested keywords & helper functions

Created on Fri Aug 11 12:48:50 2017

@author: shofer
"""
#TODO: (or maybe just N.B.) I had to place the nested keywords  outside of 
#nems.keyword because I couldn't get nest_helper to search a same-level package
#for function names. ---njs, August 11 2017


import nems.keyword as nk
import pkgutil as pk

def nested20(stack):
    """
    Keyword for 20-fold nested crossvalidation. Uses 5% validation chunks. 
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=20
    stack.valfrac=0.05
    nest_helper(stack)
        
def nested10(stack):
    """
    Keyword for 10-fold nested crossvalidation. Uses 10% validation chunks.
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=10
    stack.valfrac=0.1
    nest_helper(stack)
    
def nested5(stack):
    """
    Keyword for 10-fold nested crossvalidation. Uses 10% validation chunks.
    
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    stack.nests=5
    stack.valfrac=0.2
    nest_helper(stack)


def nest_helper(stack):
    """
    Helper function for implementing nested crossvalidation. Essentially sets up
    a loop with the estimation part of fit_single_model inside. 
    """
    #TODO: this should be updated so that the keywords are only imported on the 
    #first evaluation of the function, and not every time. This isn't a huge deal,
    #since importing doens't take that long, but it is still inefficient 
    #   ---njs, August 11 2017
    
    stack.cond=False
    while stack.cond is False:
        print('Nest #'+str(stack.cv_counter))
        stack.clear()
        stack.valmode=False
        for k in stack.keywords[0:-1]:
            for importer, modname, ispkg in pk.iter_modules(nk.__path__):
                try:
                    f=getattr(importer.find_module(modname).load_module(modname),k)
                    break
                except:
                    pass
            f(stack)
        
        stack.cv_counter+=1
        

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:27:50 2018

@author: hellerc
"""

# Functions to perform dimensionality reductions on response matrices

import numpy as np


def PCA(r, center=True):
    
    try: 
        bincount=r.shape[0]; repcount=r.shape[1]; stimcount=r.shape[2];cellcount=r.shape[3]; 
        r_pca = r.reshape(bincount*repcount*stimcount, cellcount);
    # if trial average data
    except: 
        bincount=r.shape[0]; stimcount=r.shape[1];cellcount=r.shape[2];  
        r_pca = r.reshape(bincount*stimcount, cellcount);
    
    if center is True:
        for i in range(0,cellcount):
            m = np.mean(r_pca[:,i])
            r_pca[:,i]=(r_pca[:,i]-m);
        
    U,S,V = np.linalg.svd(r_pca,full_matrices=False)
    v = S**2
    step = v;
    var_explained = []
    for i in range(0, cellcount):
        var_explained.append(100*(sum(v[0:(i+1)])/sum(v)));
    loading = V.T;
    pcs = U;
    return pcs, var_explained, step, loading
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 15:03:07 2017

@author: shofer
"""

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
import math as mt
import scipy.io
import scipy.signal
from nems_mod2 import *
from newNEM import *
import math as mt
import copy



datapath='/auto/users/shofer/data/batch294/'
est_files=[datapath + 'zee019e-b1_nat_export']

#These next 4 lines are simply importing the data into the program,
#and should be replaced with a more elegant method once the daq format
#is improved
stack=nems_stack()
stack.append(load_pupil_mat(est_files=est_files,fs=100))
stack.eval()
out1=stack.output()

respdata=copy.deepcopy(out1[0]['resp'])
sing=respdata[:,0,0]
xr=np.array(list(range(0,10)))


def raster_data(data):
    s=data.shape
    x=np.zeros((s[2],s[0]*s[1]))
    y=np.zeros((s[2],s[0]*s[1]))
    for i in range(0,s[2]):
        su=0
        for j in range(0,s[1]):
            y[i,su:(su+s[0])]=(j+1)*np.clip(data[:,j,i],0,1)
            x[i,su:(su+s[0])]=np.array(list(range(0,s[0])))
            su+=s[0]
        for n,m in enumerate(y[i,:]):
            if m==0:
                y[i,n]=None
    return(x,y)

rastx,rasty=raster_data(respdata)
plt.figure(1)
plt.scatter(rastx[0],rasty[0],s=(0.5*np.pi)*2)
plt.figure(2)
plt.scatter(rastx[1],rasty[1],s=(0.5*np.pi)*2)




'''
stimdata=copy.deepcopy(out1[0]['stim'])
stimdata=np.transpose(stimdata,(0,2,1)) #this transposes the stim data back to the README format
respdata=copy.deepcopy(out1[0]['resp'])
stimrespdata={'stim':stimdata,'resp':respdata}
s=respdata.shape
print(respdata.shape)
reshaped=np.reshape(respdata,(s[0]*s[1],s[2]),order='F') #reshapes response data correctly
print(reshaped.shape)


datadata=FERReT(stimrespdata,n_coeffs=10)

datadata.data_resample(Hz=300,newHz=100)
datadata.create_datasets(valsize=0.1)

print(datadata.train['stim'].shape)
print(datadata.val['stim'].shape)
print(datadata.train['resp'].shape)
print(datadata.val['resp'].shape)'''









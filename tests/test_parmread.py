#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 10:11:40 2018

@author: svd
"""

import io
import re
import numpy as np


def mat2py(s):
    
    s3=re.sub(r';', r'', s.rstrip())
    s3=re.sub(r'%',r'#',s3)
    s3=re.sub(r'\\',r'/',s3)
    s3=re.sub(r"\.([a-zA-Z0-9]+)'",r"XX\g<1>'",s3)
    s3=re.sub(r'globalparams\(1\)',r'globalparams',s3)
    s3=re.sub(r'exptparams\(1\)',r'exptparams',s3)
              
    s4=re.sub(r'\(([0-9]*)\)', r'[\g<1>]', s3)
#    s4=re.sub(r"\.(?m')(?evp')", r"XXXX", s4)
#    s4=re.sub(r"\.(?m')(?evp')", r"XXXX", s4)
    s5=re.sub(r'\.([A-Za-z][A-Za-z1-9_]+)', r"['\g<1>']", s4)
    
    s6=re.sub(r'([0-9]+) ', r"\g<0>,", s5)
    s6=re.sub(r'NaN ', r"np.nan,", s6)
    
    s7=re.sub(r"XX([a-zA-Z0-9]+)'",r".\g<1>'",s6)
    #s7=re.sub(r"XXevp'",r".evp'",s7)
    
    s7=re.sub(r'NaN',r'np.nan',s7)
    s7=re.sub(r'zeros\(([0-9,]+)\)',r'np.zeros([\g<1>])',s7)
    s7=re.sub(r'{(.*)}',r'[\g<1>]',s7)
    
    return s7


filepath='/Users/svd/Documents/current/nems/BRT007c05_a_PTD.m'

f = io.open(filepath, "r")

s=f.readlines(-1)

globalparams={}
exptparams={}
exptevents={}

for ts in s:
    sout=mat2py(ts)
    #print(sout)
    try:
        exec(sout)
    except KeyError:
        ts1=sout.split('= [')
        ts1=ts1[0].split(',[')
        
        s1=ts1[0].split('[')
        sout1="[".join(s1[:-1]) + ' = {}'
        try: 
            exec(sout1)
        except :
            s2=sout1.split('[')
            sout2="[".join(s2[:-1]) + ' = {}'
            exec(sout2)
            exec(sout1)
        exec(sout)
    except NameError:
        print("NameError on: {0}".format(sout))
    except:
        print("Other error on: {0} to {1}".format(ts,sout))
    
#    while not success:
        



#s2= s[4214].rstrip()
#s2="exptparams(1).Performance(97).LickRate = 0.000247831474597274;  % comment"




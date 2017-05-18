#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 10:32:04 2017

@author: jacob
"""

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import matplotlib as mpl
import matplotlib.pyplot as pl
import pymysql as pysql
import scipy as scpy


testConnection = pysql.connect(user='lbhbread',passwd='ferret33',\
                                host='hyrax.ohsu.edu',database='cell')

testCursor = testConnection.cursor()


""" testing DataFrame --- want data from cellDB to look something like this eventually
testFrame = pd.DataFrame({'cell1' : pd.Series([1.,2.,3.], index = ['r value','some other value', 'so much values']),\
                          'cell2' : pd.Series([1.,2.,3.], index = ['r value','some other value', 'so much values']),\
                          'cell3': pd.Series([1.,2.,3.], index = ['r value','some other value', 'so much values']),\
                          'cell4': pd.Series([1.,2.,3.], index = ['r value','some other value', 'so much values'])})

print(testFrame.T)
"""

#other method below seems simpler
"""
datadump = testCursor.execute("SELECT *...")

using the execute command above
dbFrame = pd.DataFrame(datadump.fetchall())
dbFrame.columns = datadump.keys()
"""


#with pandas using pysql's built-in .read_sql function. still have 
#to use pymysql to connect to database etc
#this just copies the entire table into a dataframe

print(" \n \n \n  new run  \n \n \n")
dbFrame2 = psql.read_sql("SELECT DISTINCT * FROM sCellFile LIMIT 20", testConnection).set_index('cellid')
#view top few rows of resulting dataframe to see if it worked
print(dbFrame2.head(3))
#print out one entire column (i.e. all n-params values, or all r-ceiling values, etc)
#print(dbFrame2[:10])
#print(dbFrame2['id'])

dbFrame3 = psql.read_sql("SELECT DISTINCT * FROM gSingleCell LIMIT 20", testConnection).set_index('id')
print(dbFrame3.head(3))
#print(dbFrame3[:10])

dbFrame4 = psql.read_sql("SELECT DISTINCT * FROM gData LIMIT 20", testConnection).set_index('id')
print(dbFrame4.head(3))
"""

resultsFrame = psql.read_sql('SELECT DISTINCT * FROM NarfResults WHERE n_parms IS NOT NULL LIMIT 1000', testConnection)
batchesFrame = psql.read_sql("SELECT DISTINCT * FROM NarfBatches LIMIT 100", testConnection)
analysisFrame = psql.read_sql("SELECT DISTINCT * FROM NarfAnalysis LIMIT 100", testConnection)

"""
"""
#performance scatterplot?

rvalues1 = []
rvalues2 = []


celllist1 = resultsFrame['cellid', resultsFrame.modelname == 'env100_log2b_firn_npnl_sb_mse']
celllist2 = resultsFrame['cellid', resultsFrame.modelname == 'env100_log2b_firn_npnl_fminlsq_mse']


for cell in celllist1:
    rvalue = resultsFrame['r_ceiling', resultsFrame.cellid == cell]
    rvalues1 += rvalue
    
for cell in celllist2:
    rvalue = resultsFrame['r_ceiling', resultsFrame.cellid == cell]
    rvalues2 += rvalue
    
pl.plot(rvalues1,rvalues2)

"""


testConnection.close()
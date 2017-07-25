#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 19 09:15:56 2017

@author: jacob
"""

#messing around with bokeh
from bokeh.plotting import *
import pandas as pd
import pandas.io.sql as psql
import pymysql as pysql

"""
#this should work the same way for plotting columns of a celldb table
data = pd.DataFrame({'One':pd.Series([1,2,3,4,5]), 'Two':pd.Series([1,4,11,64,3])})
p = figure()
p.line(list(data['One']),list(data['Two']), line_width=2)
show(p)
"""

CONNECTION = pysql.connect(user='lbhbread',passwd='ferret33',\
                                host='hyrax.ohsu.edu',database='cell')

def read_table():
    data = psql.read_sql('SELECT * FROM NarfResults WHERE (r_test > 0) LIMIT 100',CONNECTION)
    return data

"""
# basic line plot of id versus r_test
data = read_table()
p = figure()
p.line(list(data['id']),list(data['r_test']), line_width=2)
show(p)
"""

# scatter plot for two models' r_test performance over a batch of cells

def query_batch():
    data = psql.read_sql('SELECT * FROM NarfResults WHERE (batch=271) AND ((modelname="fb18ch100_lognn_wcg03_ap3z1_dexp_fit05v") OR (modelname="fb18ch100_lognn_wcg03_voltp_ap3z1_dexp_fit05v"))', CONNECTION)
    return data

data = query_batch()
p = figure()

modelOne = "fb18ch100_lognn_wcg03_ap3z1_dexp_fit05v"
modelTwo = "fb18ch100_lognn_wcg03_voltp_ap3z1_dexp_fit05v"
#make lists of cellid, modelname and r_test
cellList = list(data['cellid'])
print(modelone)
rList = list(data['r_test'])
#make a blank list to store the r_test values for each cell for the given model
modelOneList = []
modelTwoList = []
#loop through list of cells. indexes should all match, so just check for modelname
#match at same index. if it matches one of the two to be tested, put the associated
#r value in the appropriate list

#this code isn't really doing what I want at the moment, needs to
#only plot values for cells that have a value for both models
#but it still demonstrates the basic concept for doing the scatter plot with bokeh

"""
for cell in range(len(cellList)):
    if modelList[cell] in modelOne:
        modelOneList.append(rList[cell])
    if modelList[cell] in modelTwo:
        modelTwoList.append(rList[cell])

p.circle(modelOneList, modelTwoList, size=5, color="navy", alpha=0.5)
show(p)
"""

#Currently running VERY slow compared to narf_analysis
#will need to either find a way to speed it up or switch to some other plotting
#setup like matplotlib

CONNECTION.close()
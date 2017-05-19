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

data = read_table()
p = figure()
p.line(list(data['id']),list(data['r_test']), line_width=2)
show(p)

CONNECTION.close()

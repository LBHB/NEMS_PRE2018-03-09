#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 09:52:01 2017

@author: jacob
"""

import numpy as np
import pandas as pd
import pandas.io.sql as psql
import matplotlib as mpl
import matplotlib.pyplot as pl
import pymysql as pysql
import scipy as scpy

#connect to celldb using pymysql
celldbConnection = pysql.connect(user='lbhbread',passwd='ferret33',\
                                host='hyrax.ohsu.edu',database='cell')

celldbCursor = celldbConnection.cursor()

#use pandas.io.sql to read tables into pandas dataframes then print some stuff out
dbFrame1 = psql.read_sql("SELECT DISTINCT * FROM NarfResults WHERE n_parms IS NOT NULL LIMIT 100",\
                         celldbConnection)
print(dbFrame1.head(10))

dbFrame2 = psql.read_sql("SELECT DISTINCT * FROM sCellFile LIMIT 20", celldbConnection)
#view top few rows of resulting dataframe to see if it worked
print(dbFrame2.head(3))

dbFrame3 = psql.read_sql("SELECT DISTINCT * FROM gSingleCell LIMIT 20", celldbConnection)
print(dbFrame3.head(3))
#print(dbFrame3[:10])

dbFrame4 = psql.read_sql("SELECT DISTINCT * FROM gData LIMIT 20", celldbConnection)
print(dbFrame4.head(3))
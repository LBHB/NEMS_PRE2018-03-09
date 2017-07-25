#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 16:43:26 2017

@author: jacob
"""

import pandas as pd
import pandas.io.sql as psql
import pymysql as pysql
from flask import *
import pandas as pd

LIMIT = 2000
CONNECTION = pysql.connect(user='lbhbread',passwd='ferret33',\
                                host='hyrax.ohsu.edu',database='cell')

def read_table(tablename):
    data = psql.read_sql('SELECT * FROM %s LIMIT %d' %(tablename, LIMIT),\
                         CONNECTION)
    return data

def query_batch(batchid):
    data = psql.read_sql('SELECT * FROM NarfBatches WHERE batch=%s LIMIT %d'\
                         %(batchid, LIMIT), CONNECTION)
    return data

def query_model(modelname):
    data = psql.read_sql('SELECT * FROM NarfResults WHERE modelname="%s" LIMIT %d'\
                         %(modelname, LIMIT), CONNECTION)
    return data

def query_model_from_batch(batchid, modelname):
    data = psql.read_sql\
    ('SELECT * FROM NarfResults WHERE (batch=%s) AND (modelname="%s") LIMIT %d' \
     %(batchid, modelname, LIMIT), CONNECTION)
    return data

def close_connection():
    CONNECTION.close()
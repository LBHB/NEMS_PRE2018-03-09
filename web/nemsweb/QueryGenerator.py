"""
- Query Generator object
- Forms SQL queries from variables passed from views.py
- Then queries celldb and returns a pandas dataframe populated with the data
"""

import pandas as pd
import pandas.io.sql as psql
import pymysql as pysql
from flask import *
import pandas as pd


# global variable to limit number of rows pulled
# used as default unless user specifies another value
LIMIT = 2000
        
class QueryGenerator():
    
    def __init__(self,connection,distinct=False,column='*',tablename="NarfResults",\
                 batchnum="", modelname="",analysis="", limit=LIMIT):
        
        # always establish connection to celldb
        self.connection = connection
        self.column = column
        self.distinct = distinct
        # TODO: list out all variables and include them in arguments to initializer
        # so that they can be easily passed by views.py
        self.tablename = tablename
        self.batchnum = batchnum
        self.modelname = modelname
        # for use with NarfAnalysis table
        self.analysis = analysis
        self.limit = limit
        # generate query automatically - no reason to leave it blank
        self.query = self.generate_query()
        
    def generate_query(self):
        # TODO: for each attribute, add some constraint to query unless
        # attribute was left blank/empty, in order of MYSQL syntax
        # ex: self.query += 'tableName'
        # may need to set up different if:then routes based on tablename
        # (or a different subclass for each table or category of tables?)
        # if number of variables between all of the tables gets too complicated
        
        filterCount = 0
        
        if self.distinct:
            distinct = 'DISTINCT'
        else:
            distinct = ''
        
        q = 'SELECT %s %s FROM '%(distinct, self.column)
        q += self.tablename
        if self.batchnum != "":
            filterCount += 1
            if filterCount == 1:
                q+= ' WHERE batch=' + self.batchnum
            else:
                q+= ' AND batch=' + self.batchnum
        if self.modelname != "":
            filterCount += 1
            if filterCount ==1:
                q+= ' WHERE modelname="' + self.modelname + '"'
            else:
                q+= ' AND modelname="' + self.modelname + '"'
        
        if self.analysis != "":
            filterCount += 1
            if filterCount ==1:
                q+= ' WHERE name="' + self.analysis + '"'
            else:
                q+= ' AND name="' + self.analysis + '"'
        
        # apply row pull limit -- keep this in place while testing
        # to keep load times down
        q += (' LIMIT %d' %LIMIT)
        
        # after running through if:then for each attribute,
        # q should be a complete MYSQL query
        return q
    
    def send_query(self):
        # populate pandas dataframe with results of query generated above
        data = psql.read_sql(self.query,self.connection)
        return data
    
    
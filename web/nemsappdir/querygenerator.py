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

class QueryGenerator():
    
    def __init__(self):
        # always establish connection to celldb
        self.connection = pysql.connect(user='lbhbread',passwd='ferret33',\
                                host='hyrax.ohsu.edu',database='cell')
        
        # TODO: list out variables and include them in arguments to initializer
        # so that they can be easily passed by views.py
        
        # generate query automatically - no reason to leave it blank
        self.query = self.generate_query()
        
    def generate_query(self):
        # TODO: for each attribute, add some constraint to query unless
        # attribute was left blank/empty, in order of MYSQL syntax
        # ex: self.query += 'tableName'
        self.query = 'SELECT * FROM '
        
        
    def send_query(self):
        # populate pandas dataframe with results of query generated above
        data = psql.read_sql(self.query,self.connection)
        return data
    
    def close_connection(self):
        # call this to close database connection when no longer needed
        self.connection.close()
    
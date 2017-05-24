"""
- establishes database connection
- and closes it when app is done
"""

import pymysql as pysql

#causing issues?
import config

class DB_Connection():
    
    def __init__(self):
        pass
    
    def connect_lab(self):
        lab = config.LAB_DATABASE
        self.connection = pysql.connect(user=lab['user'],passwd=lab['passwd'],\
                                        host=lab['host'],database=lab['database'])
   
    def connect_remote(self):
        remote = config.REMOTE_DATABASE
        self.connection = pysql.connect(user=remote['user'],passwd=remote['passwd'],\
                                        host=remote['host'],database=remote['database'])
        
    def close_connection(self):
        # call this to close database connection when no longer needed
        # always connect to celldb when connection started
        #dbserver='hyrax.ohsu.edu';
        dbserver='localhost';
        self.connection = pysql.connect(user='david',passwd='nine1997',\
                                        host=dbserver,database='cell')
        
    def close_connection(self):
    # call this to close database connection when no longer needed
        self.connection.close()

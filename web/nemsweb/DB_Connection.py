"""
- establishes database connection
- and closes it when app is done
"""

import pymysql as pysql


class DB_Connection():
    def __init__(self):
    # always connect to celldb when connection started
        #dbserver='hyrax.ohsu.edu';
        dbserver='localhost';
        self.connection = pysql.connect(user='david',passwd='nine1997',\
                                        host=dbserver,database='cell')
        
    def close_connection(self):
    # call this to close database connection when no longer needed
        self.connection.close()

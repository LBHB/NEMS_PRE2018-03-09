"""
- establishes database connection
- and closes it when app is done
"""

import pymysql as pysql


class DB_Connection():
    def __init__(self):
    # always connect to celldb when connection started
        self.connection = pysql.connect(user='lbhbread',passwd='ferret33',\
                                        host='hyrax.ohsu.edu',database='cell')
        
    def close_connection(self):
    # call this to close database connection when no longer needed
        self.connection.close()
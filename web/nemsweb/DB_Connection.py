"""
- establishes database connection
- and closes it when app is done
"""

import pymysql as pysql

class DB_Connection():
    
    def __init__(self):

        # read connection info from text file
        db = {}
        with open("database_info.txt","r") as f:
            for line in f:
                key,val = line.split()
                db[key] = val
        
        self.connection = pysql.connect(user=db['user'],passwd=db['passwd'],\
                                        host=db['host'],database=db['database'])
    
    def close_connection(self):
    # call this to close database connection when no longer needed
        self.connection.close()

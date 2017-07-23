#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

nems_db library

Created on Fri Jun 16 05:20:07 2017

@author: svd
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base


DEBUG = True

db = {}
with open ("instance/database_info.txt","r") as f:
    for line in f:
        key,val = line.split()
        db[key] = val

SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://%s:%s@%s/%s'%(db['user'],db['passwd'],db['host'],\
                                                 db['database'])

# sets how often sql alchemy attempts to re-establish connection engine
# TODO: query db for time-out variable and set this based on some fraction of that
POOL_RECYCLE = 7200;

# create a database connection engine
engine = create_engine(SQLALCHEMY_DATABASE_URI,pool_recycle=POOL_RECYCLE)


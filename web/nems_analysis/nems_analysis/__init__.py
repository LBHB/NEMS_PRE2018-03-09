from flask import Flask
from sqlalchemy import create_engine
#from flask_pymemcache import FlaskPyMemcache
import pandas.io.sql as psql

app = Flask(__name__)
app.config.from_object('config')

# create a database connection engine
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])

# TODO: How to filter some of this? probably need all of narfanalysis, but
# don't use a lot of the columns of results and batches very often. Could probably
# leave them off and add on later if user chooses to enable extra details

# TODO: better way to do this using memcache? 
analyses = psql.read_sql('SELECT * FROM NarfAnalysis',engine)
batches = psql.read_sql('SELECT DISTINCT batch FROM NarfAnalysis',engine)
cells = psql.read_sql('SELECT cellid,batch FROM NarfBatches',engine)


#this doesn't get used for anything, just has to be loaded when
#app is initiated
import nems_analysis.views

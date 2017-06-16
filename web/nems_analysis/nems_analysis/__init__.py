from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

app = Flask(__name__)
app.config.from_object('config')

# sets how often sql alchemy attempts to re-establish connection engine
# TODO: query db for time-out variable and set this based on some fraction of that
POOL_RECYCLE = 7200;

#create base class to mirror existing database schema
Base = automap_base()
# create a database connection engine
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'],pool_recycle=POOL_RECYCLE)
Base.prepare(engine, reflect=True)

NarfAnalysis = Base.classes.NarfAnalysis
NarfBatches = Base.classes.NarfBatches
NarfResults = Base.classes.NarfResults
tQueue = Base.classes.tQueue
sCellFile = Base.classes.sCellFile

Session = sessionmaker(bind=engine)

#these don't get used for anything, just have to be loaded when
#app is initiated
import nems_analysis.views
import plot_functions.views
import model_functions.views
import modelpane.views

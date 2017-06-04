from flask import Flask
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

app = Flask(__name__)
app.config.from_object('config')

#create base class to mirror existing database schema
Base = automap_base()
# create a database connection engine
engine = create_engine(app.config['SQLALCHEMY_DATABASE_URI'])
Base.prepare(engine, reflect=True)

NarfAnalysis = Base.classes.NarfAnalysis
NarfBatches = Base.classes.NarfBatches
NarfResults = Base.classes.NarfResults

Session = sessionmaker(bind=engine)

#this doesn't get used for anything, just has to be loaded when
#app is initiated
import nems_analysis.views
import plot_functions.views

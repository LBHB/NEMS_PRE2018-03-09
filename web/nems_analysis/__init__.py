import sys
from io import StringIO
#import datetime

from flask import Flask
from flask_socketio import SocketIO
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.automap import automap_base

from nems_analysis.SplitOutput import SplitOutput

app = Flask(__name__)
app.config.from_object('config')

socketio = SocketIO(app, async_mode='threading')
thread = None

stringio = StringIO()
orig_stdout = sys.stdout
sys.stdout = SplitOutput(stringio, orig_stdout)

def py_console():
    while True:
        # Set sampling rate for console reader in seconds
        socketio.sleep(0.5)
        try:
            data = stringio.getvalue()
            lines = data.split('\n')
            stringio.truncate(0)
            for line in lines:
                if line:
                    # adds timestamp, which is nice, but messes with line break
                    #now = datetime.datetime.now()
                    #line = '{0}:{1}:{2}: {3}'.format(
                    #            now.hour, now.minute, now.second, line,
                    #            )
                    socketio.emit(
                            'console_update',
                            {'data':line},
                            namespace='/py_console',
                            )
        except Exception as e:
            print(e)
            pass

@socketio.on('connect', namespace='/py_console')
def start_logging():
    global thread
    if thread is None:
        print('Initializing console reader...')
        thread = socketio.start_background_task(target=py_console)
    socketio.emit(
            'console_update',
            {'data':'console connected or re-connected'},
            namespace='/py_console',
            )

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

# these don't get used for anything within this module, 
# just have to be loaded when app is initiated
import nems_analysis.views
import misc_views.views
import plot_functions.views
import model_functions.views
import modelpane.views
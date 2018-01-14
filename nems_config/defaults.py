""" Default configurations for nems_config. Should fall back on these when
a config file is not present (i.e. if Storage_Config.py doesn't exist, import
nems_config.defaults.STORAGE_DEFAULTS instead).

Also stores defaults for interface options, such as which columns to show
on the results table or what minimum SNR to require for plots by default.

"""

import sys
import logging
# Used 'root' instead of __name__ because the pre-configuration logging
# would not show up otherwise.
# And then root stopped showing up too...
# TODO: Figure out best way to get log statements in this module
#       to show up with the others.
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s : %(levelname)s : %(name)s, "
                              "line %(lineno)s:\n%(message)s\n")
ch = logging.StreamHandler(stream=sys.stdout)
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
log.addHandler(ch)

from pathlib import Path
import importlib
import os
import warnings
import traceback
import errno
import logging.config
import datetime as dt
import boto3
from botocore import UNSIGNED
from botocore.config import Config

import nems_sample as ns
import nems_logs
SAMPLE_PATH = os.path.dirname(os.path.abspath(ns.__file__))
NEMS_LOGS_PATH = os.path.dirname(os.path.abspath(nems_logs.__file__))
# stays false unless changed by db.py if database info is missing
DEMO_MODE = False

class UI_OPTIONS():
    cols = ['r_test', 'r_fit', 'n_parms']
    rowlimit = 500
    sort = 'cellid'
    # specifies which columns from narf results can be used to quantify
    # performance for plots
    measurelist = [
            'r_test' , 'r_ceiling', 'r_fit', 'r_active', 'mse_test',
            'mse_fit', 'mi_test', 'mi_fit', 'nlogl_test',
            'nlogl_fit', 'cohere_test', 'cohere_fit',
            ]
    # any columns in this list will always show on results table
    required_cols = ['cellid', 'modelname']
    # specifies which columns to display in 'details' tab for analysis
    detailcols = ['id', 'status', 'question', 'answer']
    # default minimum values for filtering out cells before plotting
    iso = 0
    snr = 0
    snri = 0

class STORAGE_DEFAULTS():
    DIRECTORY_ROOT = SAMPLE_PATH
    USE_AWS = False

class FLASK_DEFAULTS():
    Debug = False
    COPY_PRINTS = False
    CSRF_ENABLED = True

class LOGGING_DEFAULTS():
    """ Specifies default configuration for nems logging. To change settings
    for a local configuration, place a file named "Logging_Config.py" in the
    nems_config directory with the same structure as this class. Alternatively,
    if a different log file is desired but all  other settings can remain the
    same, the filepath and/or log name can be specified in the NEMSLOG and
    NEMSLOGPATH environment variables. Similarly, the logging levels for
    the console and file handlers can be overriden with variables
    NEMSLVLCON and NEMSLVLFILE, respectively.

    Order of precendence when settings conflict between environment variables,
    top-level config file variables, and dictionary values:
        1st: Environment Variables
        2nd: Top-level Variables
        3rd: Dictionary Value

    Example Logging_Config contents:
        Copy & paste the whole class, then make tweaks as desired
        (log_root doesn't have to be included if the default is acceptable,
        but keeping the full logging_config dict as a scaffold is recommended):
            log_root = 'my/file/path'

            logging_config = {
            'version': 1,
            'formatters': {
                    'basic': {'format': '}
                    'my_formatter': {'format': '%(asctime)s -- %(message)s'},
                    },
            'handlers': {
                    'console':
                        ... [abbreviated],
                        'formatter': 'my_formatter',
                    },
            'loggers': {
                    ... [abbreviated]
                    },
            'root': {
                    'handlers': ['console'],
                    },
            }

    Example environment variable specifications:
    #1) Specify exact file name:
        NEMSLOG='/my/file/path/log_name.log'
        export NEMSLOG
        nems-fit-single . . .

    #2) Specify directory root, but let nems decide the file name:
        NEMSLOGPATH='/my/file/path/'
        export NEMSLOGPATH
        nems-fit-single . . .

    #3) Specify different log levels for the console and file handlers:
        NEMSLVLFILE='INFO'
        NEMSLVLCON='WARNING'
        export NEMSLVLFILE NEMSLVLCON
        nems-fit-single . . .

    NOTES: -If NEMSLOG is specified, it will override NEMSLOGPATH.
           -The top-level log_root variable overrides any filename specified
            within the config dictionary, so it should be reassigned to a blank
            string (i.e. '') if a filename specified in the dictionary
            is desired.

    """

    # directory to store log files in
    log_root = NEMS_LOGS_PATH
    # log levels for the respective handlers
    # note: these levels will override values in the dictionary if specified.
    console_level = 'INFO'
    file_level = 'DEBUG'

    logging_config = {
            'version': 1,
            'formatters': {
                    'basic': {'format': (
                                "%(asctime)s : %(levelname)s : %(name)s, "
                                "line %(lineno)s:\n%(message)s\n"
                                )
                        },
                    'short': {'format': "%(name)s, %(lineno)s : %(message)s\n"},
                    },
            'handlers': {
                    # only console logger included by default,
                    # log file added at runtime if filename is present
                    # (a default filename will be present unless overwritten)
                    'console': {
                            'class': 'logging.StreamHandler',
                            'formatter': 'short',
                            'level': 'INFO',
                            },
                    'file': {
                            'class': 'logging.FileHandler',
                            'formatter': 'basic',
                            'encoding': 'UTF-8',
                            'filename': '',
                            'mode': 'a',
                            # 0: none, 10: DEBUG, 20: INFO, 30: WARNING,
                            # 40: ERROR, 50: CRITICAL
                            'level': 'DEBUG',
                            }
                    },
            'loggers': {
                    # This level should be left at DEBUG;
                    # to adjust message display, change logging level for
                    # console and/or file handlers as needed
                    'nems': {'level': 'DEBUG'},
                    'nems_config': {'level': 'DEBUG'},
                    'nems_scripts': {'level': 'DEBUG'},
                    'nems_web': {'level': 'DEBUG'},
                    },
            'root': {
                    'handlers': ['console'],
                    },
            }

def update_settings(module_name, default_class):
    """ Overwrites contents of default_class with contents of the specified
    config module. Only attributes specified in the config module will be
    overwritten, others are left as defaults."""

    try:
        log.debug("Attempting to update {0} with values from {1}"
                  .format(default_class, module_name))
        mod = importlib.import_module('.' + module_name, 'nems_config')
    except Exception as e:
        log.debug("Error when attempting to import settings for {0}: {1}"
                  .format(e, module_name))
        log.debug("Couldn't import settings for: {0} -- using defaults... "
                  .format(module_name))
        return

    for key in mod.__dict__:
        setattr(default_class, key, getattr(mod, key))

# update config classes with variables specified in user-provided
# config modules.
update_settings("Storage_Config", STORAGE_DEFAULTS)
update_settings("Flask_Config", FLASK_DEFAULTS)
update_settings("Logging_Config", LOGGING_DEFAULTS)

def configure_logging():
    logging_config = LOGGING_DEFAULTS.logging_config
    console_level = LOGGING_DEFAULTS.console_level
    file_level = LOGGING_DEFAULTS.file_level
    log_root = LOGGING_DEFAULTS.log_root
    # filename should be an empty string unless added by user
    filename = logging_config['handlers']['file']['filename']

    # filename specified by environment variable will take precedence
    try:
        filename = os.environ['NEMSLOG']
    except:
        log.debug("No filename specified in OS environment, trying file"
                  "specified in Logging_Config or LOGGING_DEFAULTS")
        try:
            # root directory in environment variables takes precedence
            log_root = os.environ['NEMSLOGPATH']
        except:
            log.debug("No fileroot specified in OS environment, trying"
                      "path specified in Logging_Config or LOGGING_DEFAULTS")
        if log_root:
            # if log_root wasn't removed by user, set log file to
            # time-based filename inside that directory
            timestamp = dt.datetime.now().strftime('%Y-%m-%d')
            file = 'nemslog_{0}.log'.format(timestamp)
            filename = os.path.join(log_root, file)
            try:
                os.makedirs(log_root)
            except OSError as e:
                # check if error was because directory already exists
                if e.errno != errno.EEXIST:
                    log.debug("Log file directory could not be created: {0} --"
                              "\nError: {1}"
                              .format(log_root, e))
                    filename = None
                else:
                    pass

    if filename:
        # if log_root wasn't specified, but a filename was, try
        # creating directory from filename instead.
        # find the last / in filename, and chop off the rest to get root
        root_idx = filename.rfind('/')
        file_root = filename[:root_idx+1]
        try:
            os.makedirs(file_root)
        except OSError as e:
            if e.errno != errno.EEXIST:
                log.debug("Log file directory could not be created: {0} --"
                          "\nError: {1}"
                          .format(file_root, e))
                filename = None
            else:
                pass

    # levels specified in environment variables will take precedence
    try:
        console_level = os.environ['NEMSLVLCON']
    except:
        pass
    try:
        file_level = os.environ['NEMSLVLFILE']
    except:
        pass

    log.debug("log filename ended up being: %s"%filename)
    # if file handler was'nt removed by user and filename isn't blank,
    # configure the relevant keys and add the file handler to the root.
    if filename and 'file' in logging_config['handlers']:
        logging_config['handlers']['file']['filename'] = filename
        logging_config['handlers']['file']['level'] = file_level
        logging_config['root']['handlers'].append('file')
    logging_config['handlers']['console']['level'] = console_level
    log.debug("logging_config dict ended up being: %s"%logging_config)
    logging.config.dictConfig(logging_config)

try:
    configure_logging()
    log.debug("logging successfully configured.")
    # reconfigure logger for this module after settings applied
    log = logging.getLogger(__name__)
except Exception as e:
    log.warning("Attempt to configure logging resulted in error: {0}"
                .format(e))

# send uncaught exceptions to log file if file handler is set up.
# will also go to console as normal.
exlogger_name = "Uncaught_Exception"
exlog = logging.getLogger(exlogger_name)
def uncaught_exception_handler(type, value, tb):
    string = "{0}, TRACEBACK:".format(type)
    hashline = "#"*(len(string)+len(exlogger_name)+2)
    exlog.exception("{0}\n{1}\n".format(string, hashline))
    tblist = traceback.format_list(traceback.extract_tb(tb))
    for i, tb in enumerate(tblist):
        exlog.exception(str(i) + ": " + tb)
sys.excepthook = uncaught_exception_handler

# TODO: add logger for warnings formatting
#import sys
#import warnings
#import traceback
#def warn_with_traceback(message, category, filename, lineno, file=None,
#                        line=None):
#    log = file if hasattr(file, 'write') else sys.stderr
#    traceback.print_stack(file=log)
#    log.write(warnings.formatwarning(message, category, filename,
#                                     lineno, line))
#warnings.showwarning = warn_with_traceback


db_path = os.path.join(SAMPLE_PATH, 'demo_db.db')
log.debug("db_path for demo ended up being: {0}".format(db_path))
db_obj = Path(db_path)
# Check if sample database exists. If it doesn't, get it from the public s3
if not db_obj.exists():
    log.info("Demo database not found, retrieving....")
    s3_client = boto3.client(
            's3',
            #aws_access_key_id='dummyid', aws_secret_access_key='dummykey',
            #aws_session_token='dummytoken',
            #config=Config(signature_version=UNSIGNED),
            )
    key = "demodb/demo_db.db"
    fileobj = s3_client.get_object(Bucket='nemspublic', Key=key)
    with open(db_path, 'wb+') as f:
        f.write(fileobj['Body'].read())
        log.info("Demo database written to: ")
        log.info(db_path)
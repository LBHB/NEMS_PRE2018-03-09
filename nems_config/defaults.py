""" Default configurations for nems_config. Should fall back on these when
a config file is not present (i.e. if Storage_Config.py doesn't exist, import
nems_config.defaults.STORAGE_DEFAULTS instead).

Also stores defaults for interface options, such as which columns to show
on the results table or what minimum SNR to require for plots by default.

"""

import logging
# Logging level for this module should only be set to debug if... well,
# if this specific code needs to be debugged. Otherwise should be kept at info
# or above to avoid excessive log statements every time this module is imported
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
#ch = logging.StreamHandler()
#ch.setLevel(logging.DEBUG)
#fm = logging.Formatter(
#        '%(asctime)s -- %(name)s --  %(levelname)s : %(message)s'
#        )
#ch.setFormatter(fm)
#log.addHandler(ch)

from pathlib import Path
import importlib
import sys
import os
import warnings
import traceback
import errno
import logging.config
import datetime as dt
import nems_sample as ns
import boto3
from botocore import UNSIGNED
from botocore.config import Config
sample_path = os.path.dirname(os.path.abspath(ns.__file__))

# stays false unless changed by db.py if database info is missing
DEMO_MODE = False
# sets to true after configure_logging call completes without error

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
    DIRECTORY_ROOT = sample_path
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
    NEMSLOGPATH environment variables.

    Example Logging_Config contents:
    #1) Copy the whole class, then make tweaks as desired:
            log_root = 'my/file/path'
            shortened_format = '%(asctime)s -- %(message)s'

            logging_config = {
            'version': 1,
            'formatters': {
                    'my_formatter': {'format': shortened_format},
                    },
            'handlers': {
                    ... [abbreviated]
                    },
            'loggers': {
                    ... [abbreviated]
                    },
            'root': {
                    'handlers': ['console'],
                    },
            }

    #2) Only specifying attributes to replace (or new ones to add):
            basic_format = '%(name)s -- %(message)s'
            new_format = '%(levelname)s -- %(message)s'
            logging_config['handlers']['console']['level'] = 'INFO'
            (unspecified values are left unchanged)

    Example environment variable specifications:
    #1) Specify exact file name:
        NEMSLOG = '/my/file/path/log_name.log'
        export NEMSLOG
        nems-fit-single . . .

    #2) Specify directory root, but let nems decide the file name:
        NEMSLOGPATH = '/my/file/path/'
        export NEMSLOGPATH
        nems-fit-single . . .

    NOTE: If NEMSLOG is specified, it will override NEMSLOGPATH

    """

    # directory to store log files in
    log_root = '~/nemslogs/'
    # formatting of logging messages
    basic_format = ("%(asctime)s : %(levelname)s : %(name)s, "
                    "line %(lineno)s:\n%(message)s\n")
    short_format = "%(name)s : %(message)s\n"

    logging_config = {
            'version': 1,
            'formatters': {
                    'basic': {'format': basic_format},
                    'short': {'format': short_format},
                    },
            'handlers': {
                    # only console logger included by default,
                    # log file added at runtime if filename is present
                    # (a default filename will be present unless overwritten)
                    'console': {
                            'class': 'logging.StreamHandler',
                            'formatter': 'short',
                            'level': 'DEBUG',
                            #'stream': 'ext://sys.stdout',
                            },
                    },
            'loggers': {
                    'nems': {'level': 'DEBUG'},
                    #'exception_logger': {'level': 'DEBUG',
                    #                     'handlers': ['console'],
                    #                     },
                    },
            'root': {
                    'handlers': ['console'],
                    },
            }

sample_i = sample_path.find('/nems_sample')
db_path = sample_path[:sample_i] + '/nems_sample/demo_db.db'
log.info(db_path)
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

# TODO: Any way to put this outside of the config file and still guarantee
#       that it gets run?
#       Can put it in app initialization for web app, but some modules use
#       these settings w/o ever launching the app.
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
    filename = None

    # filename specified by environment variable will take precedence
    try:
        filename = os.environ['NEMSLOG']
    except:
        log.debug("No filename specified in OS environment, trying file"
                  "specified in Logging_Config or LOGGING_DEFAULTS")
        try:
            log_root = os.environ['NEMSLOGPATH']
        except:
            log.debug("No fileroot specified in OS environment, trying"
                      "path specified in Logging_Config or LOGGING_DEFAULTS")
            log_root = LOGGING_DEFAULTS.log_root
        timestamp = dt.datetime.now().strftime('%Y-%m-%d')
        file = 'nemslog_{0}.log'.format(timestamp)
        filename = os.path.join(log_root, file)
        try:
            os.makedirs(log_root)
        except OSError as e:
            # check if error was because directory already exists
            if e.errno != errno.EEXIST:
                log.debug("Log file directory could not be created: {0} --"
                      "Outputting logs to console only."
                      .format(log_root))
                filename = None
            else:
                # if that was the case, no problem
                pass

    log.debug("log filename ended up being: %s"%filename)
    if filename:
        logging_config['handlers']['file'] = {
                'class': 'logging.FileHandler',
                'formatter': 'basic',
                'filename': filename,
                'encoding': 'UTF-8',
                'mode': 'a',
                # 0: none, 10: DEBUG, 20: INFO, 30: WARNING,
                # 40: ERROR, 50: CRITICAL
                'level': logging.DEBUG,
                }
        logging_config['root']['handlers'].append('file')
    log.debug("logging_config dict ended up being: %s"%logging_config)
    logging.config.dictConfig(logging_config)

try:
    configure_logging()
    LOGGING_CONFIGURED = True
    log.debug("logging successfully configured.")
except Exception as e:
    log.warn("Attempt to configure logging resulted in error: {0}".format(e))

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
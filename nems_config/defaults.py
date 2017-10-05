""" Default configurations for nems_config. Should fall back on these when
a config file is not present (i.e. if Storage_Config.py doesn't exist, import
nems_config.defaults.STORAGE_DEFAULTS instead).

Also stores defaults for interface options, such as which columns to show
on the results table or what minimum SNR to require for plots by default.

"""

import importlib
import os
import nems_sample as ns
sample_path = os.path.abspath(ns.__file__)[:-11]

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

# TODO: Any way to put this outside of the config file and still guarantee
#       that it gets run?
#       Can put it in app initialization for web app, but some modules use
#       these settings w/o ever launching the app.
def update_settings(module_name, default_class):
    try:
        mod = importlib.import_module('.' + module_name, 'nems_config')
    except Exception as e:
        print(e)
        print(
                "Couldn't import settings for: %s -- using defaults... "
                %module_name
                )
        return
    
    for key in mod.__dict__:
        setattr(default_class, key, getattr(mod, key))
        
update_settings("Storage_Config", STORAGE_DEFAULTS)
update_settings("Flask_Config", FLASK_DEFAULTS)
#!/usr/bin/python3

import sys
import nems.xforms as xforms
from  nems.urls import load_resource

def reload_model(model_uri):
    '''
    Reloads an xspec and modelspec that were saved in some directory somewhere.
    This recreates the context that occurred during the fit.
    Passes additional context {'IsReload': True}, which xforms should react to
    if they are not intended to be run on a reload.
    '''
    # uris = list_directory(model_uri)
    xfspec_uri = '/home/ivar/results/TAR010c-18-1/wc18x1_lvl1_fir15x1/fit_basic/2018-03-07T23:40:48/xfspec.json'
    modelspec_uri = '/home/ivar/results/TAR010c-18-1/wc18x1_lvl1_fir15x1/fit_basic/2018-03-07T23:40:48/modelspec.0000.json'

    xfspec = load_resource(xfspec_uri)
    modelspec = load_resource(modelspec_uri)

    print(xfspec)
    print(modelspec)

    ctx, reloadlog = xforms.evaluate(xfspec, {'IsReload': True,
                                              'modelspecs': [modelspec]})

    return ctx


def print_usage():
    print('''
Usage:
      ./reload_model.py <model_uri>

Examples of valid <model_uri>:
  http://potoroo/results/TAR010c-02-1/wc18x1_lvl1_fir15x1/None/2018-03-07T22%3A55%3A11/
  /home/ivar/results/TAR010c-18-1/wc18x1_lvl1_fir15x1/fit_basic/2018-03-07T23:40:48/
 ''')

# Parse the command line arguments and do the fit
if __name__ == '__main__':
    if len(sys.argv) != 2:
        print_usage()
    else:
        reload_model(sys.argv[1])
        print('Successfully reloaded')

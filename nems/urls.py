import logging
import requests
from nems.modelspec import get_modelspec_shortname, get_modelspec_metadata

log = logging.getLogger(__name__)

# Where the filesystem organization of nems directories are decided

# TODO: MAKE ENV VARIABLES
HOST = 'potoroo'  # Add port if needed
ROUTE = 'http://' + HOST + '/results'


def relative_path(recording, modelname, fitter, date):
    '''
    Returns a relative path (excluding filename, host, port) for URLs.
    Editing this function edits the relative path of every file saved.
    '''
    if not recording and modelname and fitter and date:
        raise ValueError('Not all necessary fields defined!')
    path = '/' + recording + '/' + modelname + '/' + fitter + '/' + date + '/'
    return path


def modelspec_basepath(modelspec):
    '''
    Returns the relative path of a modelspec.
    '''
    meta = get_modelspec_metadata(modelspec)
    # Warn if not all metadata fields were found
    for f in ['fitter', 'recording', 'date']:
        if f not in meta:
            log.warn('{} not found in metadata; using "unknown"'.format(f))
    path = relative_path(modelname=get_modelspec_shortname(modelspec),
                         recording=meta.get('recording', 'unknown'),
                         fitter=meta.get('fitter', 'unknown'),
                         date=meta.get('date', 'unknown'))
    return path


def http_put(url, data=None, json=None):
    '''
    A wrapper for an HTTP put request. Returns an error string if there
    was a problem, or None if there were no errors. Please check the returned
    value.
    '''
    if json:
        r = requests.put(url, json=json)
    elif data:
        r = requests.put(url, data=data)
    else:
        raise ValueError('data or json must be defined!')

    if r.status_code == 200:
        return None
    else:
        message = 'HTTP PUT failed. Got {}: {}'.format(r.status_code, r.text)
        log.warn(message)
        return message


def save_to_url(url):
    '''
    Loads whatever is at the URL.
    URL may be a local file, an S3 file, or whatever.
    '''


def save_modelspec_in_nemsdb(modelspec):
    baseurl = nemsdb_modelspec_basepath(modelspec)
    filename = 'modelspec.{:04d}.json'.format(number)


def save_modelspec_to_nems_db(modelspec, number):
    url = nemsdb_modelspec_url(modelspec, number=number)
    return http_put(url, json=modelspec)


def save_performance_to_nems_db(modelspec):
    url = nemsdb_modelspec_url(modelspec)
    return http_put(url, json=modelspec)


def save_log_to_nems_db(modelspec):
    url = nemsdb_modelspec_url(modelspec)
    return http_put(url, data=modelspec)


def save_to_nems_db(modelspecs, performance, log, images):
    for i, m in enumerate(modelspecs):
        save_modelspec_to_nems_db(m, number=i)
    save_performance_to_nems_db(performance)
    save_log_to_nems_db(log)
    for i in images:
        save_log_to_nems_db(log)


# def load_modelspec(recording, modelname, fitter, date):
#     url = as_url(modelname=modelname, recording=recording,
#                  fitter=fitter, date=date)
#     print("Sending get request with url: {}".format(url))
#     r = requests.get(url)
#     print("Got back json: {}".format(r))

# load_modelspec(
#         recording='TAR010c-18-1',
#         modelname='TAR010c-18-1.wc18x1_lvl1_fir15x1_dexp1.fit_basic.2018-03-04T03:32:25',
#         fitter='fit_basic',
#         date='2018-02-26T19:28:57'
#         )

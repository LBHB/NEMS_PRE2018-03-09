import json as jsonlib
import logging
import requests
import nems.utils
from nems.modelspec import get_modelspec_shortname, get_modelspec_metadata, NumpyAwareJSONEncoder

log = logging.getLogger(__name__)

# Where the filesystem organization of nems directories are decided

def _tree_path(recording, modelname, fitter, date):
    '''
    Returns a relative path (excluding filename, host, port) for URLs.
    Editing this function edits the relative path of every file saved!
    '''
    if not recording and modelname and fitter and date:
        raise ValueError('Not all necessary fields defined!')
    path = '/' + recording + '/' + modelname + '/' + fitter + '/' + date + '/'
    return path


def tree_path(modelspec):
    '''
    Returns the relative path of a modelspec.
    '''
    meta = get_modelspec_metadata(modelspec)
    # Warn if not all metadata fields were found
    for f in ['fitter', 'recording', 'date']:
        if f not in meta:
            log.warn('{} not found in metadata; using "unknown"'.format(f))
    path = _tree_path(modelname=get_modelspec_shortname(modelspec),
                      recording=meta.get('recording', 'unknown'),
                      fitter=meta.get('fitter', 'unknown'),
                      date=meta.get('date', nems.utils.iso8601_datestring()))
    return path


def http_put(url, data=None, json=None):
    '''
    A wrapper for an HTTP put request. Returns an error string if there
    was a problem, or None if there were no errors. Please check the returned
    value.
    '''
    if json:
        # Serialize and unserialize to force numpy arrays to good json here
        s = jsonlib.dumps(json, cls=NumpyAwareJSONEncoder)
        js = jsonlib.loads(s)
        r = requests.put(url, json=js)
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


def save_to_nems_db(destination,
                    modelspecs,
                    xfspec,
                    images,
                    log):
    # TODO: Ensure all modelspecs have the same organizing path
    # or this next line may save things to the wrong place
    treepath = tree_path(modelspecs[0])
    base_uri = destination + treepath
    for number, modelspec in enumerate(modelspecs):
        http_put(base_uri + 'modelspec.{:04d}.json'.format(number),
                 json=modelspec)
    for number, image in enumerate(images):
        http_put(base_uri + 'figure.{:04d}.png'.format(number),
                 data=image)
    http_put(base_uri + 'log.txt',
             data=log)
    http_put(base_uri + 'xfspec.json',
             json=xfspec)
    # TODO: check the return codes, and if there was a problem,
    # have a fallback to a local file
    return None


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
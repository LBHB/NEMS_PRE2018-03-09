import io
import os
import json as jsonlib
import logging
import requests
import numpy as np
from requests.exceptions import ConnectionError
from nems.distributions.distribution import Distribution

log = logging.getLogger(__name__)

# Where the filesystem organization of nems directories are decided,
# and generic methods for saving and loading resources over HTTP or
# to local files.


class NumpyAwareJSONEncoder(jsonlib.JSONEncoder):
    '''
    For serializing Numpy arrays safely as JSONs. From:
    https://stackoverflow.com/questions/3488934/simplejson-and-numpy-array
    '''
    def default(self, obj):
        if issubclass(type(obj), Distribution):
            return obj.tolist()
        if isinstance(obj, np.ndarray):  # and obj.ndim == 1:
            return obj.tolist()
        return jsonlib.JSONEncoder.default(self, obj)


def local_uri(uri):
    '''Returns the local filepath if it is a local URI, else None.'''
    if uri[0:7] == 'file://':
        return uri[7:]
    elif uri[0] == '/':
        return uri
    else:
        return None


def http_uri(uri):
    '''Returns the URL if it is a HTTP/HTTPS URI, else None.'''
    if uri[0:7] == 'http://' or uri[0:8] == 'https://':
        return uri
    else:
        return None


def targz_uri(uri):
    '''Returns the URI if it is a .tar.gz URI, else None.'''
    if uri[-7:] == '.tar.gz' or uri[-4:] == '.tgz':
        return uri
    else:
        return None


def tree_path(modelname='undefined',
              recording='undefined',
              fitter='undefined',
              date='undefined',
              **unused_kwargs):
    '''
    Returns a relative path (excluding filename, host, port) for URIs.
    Editing this function edits the path in the file tree of every
    file saved!
    '''
    # Warn if not all metadata fields were found
    for f in [modelname, fitter, recording, date]:
        if f == 'undefined':
            log.warn('{} is "undefined" when making treepath'.format(f))

    path = '/' + recording + '/' + modelname + '/' + fitter + '/' + date + '/'

    return path


def save_resource(uri, data=None, json=None):
    '''
    For saving a resource to a URI. Throws an exception if there was a
    problem saving.
    '''
    err = None
    if json:
        if http_uri(uri):
            # Serialize and unserialize to make numpy arrays safe
            s = jsonlib.dumps(json, cls=NumpyAwareJSONEncoder)
            js = jsonlib.loads(s)
            try:
                r = requests.put(uri, json=js)
                if r.status_code != 200:
                    err = 'HTTP PUT failed. Got {}: {}'.format(r.status_code,
                                                               r.text)
            except:
                err = 'Unable to connect; is the host ok and URI correct?'
            if err:
                log.warn(err)
                raise ConnectionError(err)
        elif local_uri(uri):
            filepath = local_uri(uri)
            # Create any necessary directories
            dirpath = os.path.dirname(filepath)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            with open(filepath, mode='w+') as f:
                jsonlib.dump(json, f, cls=NumpyAwareJSONEncoder)
                f.close()
                os.chmod(filepath, 0o666)
        else:
            raise ValueError('URI type unknown')
    elif data:
        if http_uri(uri):
            try:
                r = requests.put(uri, data=data)
                if r.status_code != 200:
                    err = 'HTTP PUT failed. Got {}: {}'.format(r.status_code,
                                                               r.text)
            except:
                err = 'Unable to connect; is the host ok and URI correct?'
            if err:
                log.warn(err)
                raise ConnectionError(err)
        elif local_uri(uri):
            filepath = local_uri(uri)
            dirpath = os.path.dirname(filepath)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            if type(data) is str:
                d = io.BytesIO(data.encode())
            else:
                d = io.BytesIO(data)
            with open(filepath, mode='wb') as f:
                f.write(d.read())
            os.chmod(filepath, 0o666)
        else:
            raise ValueError('URI type unknown')
    else:
        raise ValueError('optional args data or json must be defined!')
    return err


def load_resource(uri):
    '''
    Loads and returns the resource (probably a JSON) found at URI.
    '''
    if http_uri(uri):
        r = requests.get(uri)
        if r.status_code != 200:
            err = 'HTTP GET failed. Got {}: {}'.format(r.status_code,
                                                       r.text)
            raise ConnectionError(err)
        if r.json:
            return r.json
        else:
            return r.data
    elif local_uri(uri):
        filepath = local_uri(uri)
        with open(filepath, mode='r') as f:
            resource = f.read()
        return resource
    else:
        raise ValueError('URI resource type unknown')

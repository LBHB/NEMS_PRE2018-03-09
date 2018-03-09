import re
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
            elif type(data) is io.BytesIO:
                d = data
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
            if filepath[-5:] == '.json':
                resource = jsonlib.load(f)
            else:
                resource = f.read()
        return resource
    else:
        raise ValueError('URI resource type unknown')


LINK_MATCHER = re.compile(r'<a href="(.*?)">(.*?)</a>')


def list_targz_in_nginx_dir(uri):
    '''
    Reads NGINX directory listing at URI and returns a list of URIs that
    end in .tar.gz that were found in the HTML directory listing.

    NOTE: This is a BRITTLE HACK and should not preferred in general to
    getting a list of files from something smarter, like a database
    that manages 'batches'. Ideally, such a database would return a JSON
    containing a list of URIs.
    '''
    r = requests.get(uri)

    if r.status_code != 200:
        return None

    uris = []
    for link, file in LINK_MATCHER.findall(r.content.decode()):
        if link == file and file[-7:] == '.tar.gz':
            uris.append(os.path.join(uri, file))

    return uris

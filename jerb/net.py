""" jerbs/net.py: For sending jerbs over the network. """

import json
import requests
import jerb.util
from jerb.Jerb import Jerb, load_jerb_from_file, valid_metadata_structure
from jerb.Jerb import valid_SHA1_string

##############################################################################
# Set up default routes:

req_env_vars = ['JERB_INDEX_HOST',
                'JERB_INDEX_PORT',
                'JERB_STORE_HOST',
                'JERB_STORE_PORT']

creds = jerb.util.ensure_env_vars(req_env_vars)

JSTORE = ('http://' + creds['JERB_STORE_HOST']
          + ":" + creds['JERB_STORE_PORT']+'/')

JINDEX = ('http://' + creds['JERB_INDEX_HOST']
          + ":" + creds['JERB_INDEX_PORT']+'/')


##############################################################################
# Storing and indexing


def store_jerb(jerb, route=JSTORE+'jid/'):
    """ Stores the jerb in the jerbstore at endpoint."""
    url = route + str(jerb.jid)
    return send_jerb(jerb, url)


def index_jerb(jerb, route=JINDEX+'jid/'):
    """ Indexes the jerb for search at jerb_index at route. """
    url = route + str(jerb.jid)
    return send_jerb(jerb, url)


def send_jerb(jerb, url, method='PUT'):
    """ Transmits JERB to URL as a JSON. Method defaults to 'PUT', but
    you may also use 'POST'. Returns the result. """
    headers = {"Content-Type": "application/json"}
    result = requests.put(url, data=str(jerb), headers=headers)
    return result


def publish_jerb(jerb):
    """ Stores and indexes the jerb object. """
    store_result = store_jerb(jerb)
    index_result = index_jerb(jerb)
    return (store_result, index_result)


def publish_jerbfile(jerbpath):
    """ Stores and indexes the jerbfile on at jerbpath."""
    j = load_jerb_from_file(jerbpath)
    codes = publish_jerb(j)
    return codes


##############################################################################
# Searching for jerbs matching patterns


def valid_query_structure(query):
    """ Predicate. True when query is in the correct data format."""
    # TODO: Upgrade me later to be full-featured
    return valid_metadata_structure(query)


def find_jerbs(query, query_route=JINDEX+'find'):
    """ TODO. Returns a list of JIDs matching the query. """
    s = json.loads(query)
    if not valid_query_structure(s):
        raise ValueError('find_jerbs received invalid JSON query format.')
    headers = {"Content-Type": "application/json"}
    result = requests.post(query_route, data=query, headers=headers)
    if result.status_code == 200:
        return result.json()['jids']
    else:
        print(result.raw)
        raise ValueError("Bad HTTP status code from find_jerbs")


def fetch_jerb(jid, jerbstore_route=JSTORE+'jid/'):
    """ Fetches the jerb and returns the newly loaded object. """
    if not valid_SHA1_string(jid):
        raise ValueError('fetch_jerb received an invalid SHA1')
    url = jerbstore_route + jid
    result = requests.get(url)
    if result.status_code == 200:
        j = Jerb(result.content.decode())
        return j
    else:
        print(result.raw)
        raise ValueError("Bad HTTP Status code from fetch_jerb")


def fetch_metadata(jid, jerb_index_route=JINDEX+'jid/'):
    """ Fetches the metadata for the jerb at JID. """
    if not valid_SHA1_string(jid):
        raise ValueError('fetch_metadata received an invalid SHA1')
    url = jerb_index_route + jid
    result = requests.get(url)
    if result.status_code == 200:
        return json.loads(result.content.decode())
    else:
        print(result.raw)
        raise ValueError("Bad HTTP Status code from fetch_metadata")


def get_ref(user, branch, query_route=JINDEX+'ref'):
    """ Returns the JID found at the user/branch ref."""
    # TODO: Error checking on user/branch
    params = {'user': user, 'branch': branch}
    result = requests.get(query_route, params=params)
    if result.status_code == 200:
        d = json.loads(result.content.decode())
        if 'jids' in d:
            return d['jids']
        else:
            raise ValueError('jids not found in response')
    else:
        print(result.content)
        raise ValueError("Bad HTTP status code from get_ref")

##############################################################################
# Convenience functions: abbreviations for some of the above functions


def rootjids(user=None):
    """ Convenience. Returns all the roots of the JID tree. Add the
    optional user parameter to restrict search to just one user's roots."""
    if user:
        query = {'user': user, 'parents': None}
    else:
        query = {'parents': None}
    return find_jerbs(json.dumps(query))


def children(jid):
    """ Convenience. Returns all the children of a particular JID. """
    query = {'parents': jid}
    return find_jerbs(json.dumps(query))


def parents(jid):
    """ Convenience. Returns all the parents of a particular JID"""
    md = fetch_metadata(jid)
    return md['parents']

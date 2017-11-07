""" jerbs/net.py: For sending jerbs over the network. """

import json
import requests
from jerb.Jerb import Jerb, load_jerb_from_file, valid_metadata_structure
from jerb.Jerb import valid_SHA1_string


##############################################################################
# Storing and indexing


def store_jerb(jerb, route='http://localhost:3000/jid/'):
    """ Stores the jerb in the jerbstore at endpoint."""
    url = route + str(jerb.jid)
    send_jerb(jerb, url)


def index_jerb(jerb, route='http://localhost:3001/jid/'):
    """ Indexes the jerb for search at jerb_index at route. """
    url = route + str(jerb.jid)
    send_jerb(jerb, url)


def send_jerb(jerb, url, method='PUT'):
    """ Transmits JERB to URL as a JSON. Method defaults to 'PUT', but
    you may also use 'POST'. Returns the result. """
    headers = {"Content-Type": "application/json"}
    result = requests.put(url, data=jerb.as_json(), headers=headers)
    return result


def share_jerb(jerb):
    """ Stores and indexes the jerb object. """
    store_result = store_jerb(jerb)
    index_result = index_jerb(jerb)
    return (store_result, index_result)


def share_jerbfile(jerbpath):
    """ Stores and indexes the jerbfile on at jerbpath."""
    j = load_jerb_from_file(jerbpath)
    codes = share_jerb(j)
    return codes


##############################################################################
# Searching for jerbs matching patterns


def valid_query_structure(query):
    """ Predicate. True when query is in the correct data format."""
    # TODO: Upgrade me later to be full-featured
    valid_metadata_structure(query)


def find_jerbs(query, query_route='http://localhost:3001/find'):
    """ TODO. Returns a list of JIDs matching the query. """

    if not valid_query_structure(json.loads(query)):
        raise ValueError('find_jerbs received invalid JSON query format.')

    headers = {"Content-Type": "application/json"}
    result = requests.post(query_route, data=query, headers=headers)

    if result.status_code == 200:
        return result.json()['jids']
    else:
        print(result.raw)
        raise ValueError("Bad HTTP status code from find_jerbs")


def fetch_jerb(jid, jerbstore_route='http://localhost:3000/jid/'):
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

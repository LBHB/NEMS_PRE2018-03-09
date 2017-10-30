from base64 import b64encode
import binascii
import dateutil.parser
from functools import reduce
import hashlib
import json
import os
import zlib


def threadfirst(x_init, *args):
    """Thread x_init through multiple functions f1, f2, f3...
    Example:
    >>> threadfirst(5, lambda x : x+1, lambda x : x+2)
    8
    """
    ret = reduce(lambda x, f: f(x), args, x_init)
    return ret


def merge_dicts(x, y):
    """ Given two dicts, merge them into a new dict as a shallow copy.
    Example:
    >>> merge_dicts({'a':1, 'b':2}, {'b':3}) == {'a': 1, 'b': 3}
    True
    """
    z = x.copy()
    z.update(y)
    return z


def just_keys(dictionary, keys):
    """ Given a dictionary and a list of keys, return a new hash with
    only the keys in it, if they exist.
    Example:
    >>> just_keys({'a':1, 'b':2}, ['a', 'c']) == {'a': 1}
    True
    """
    vals = [dictionary.get(k, 'None') for k in keys]
    d = dict(zip(keys, vals))
    return d


# TODO: I should rethink this decision. This is the wrong way to do this.
def environment_credentials():
    """ Returns a dict of the credentials found in the environment. """

    default_env = {'MYSQL_PORT': '3306'}
    cred_keys = ['MYSQL_HOST',
                 'MYSQL_USER',
                 'MYSQL_PASS',
                 'MYSQL_DB',
                 'MYSQL_PORT',
                 'REDIS_HOST',
                 'REDIS_PORT',
                 'REDIS_PASS',
                 'AWS_ACCESS_KEY_ID',
                 'AWS_SECRET_KEY']
    env = merge_dicts(default_env, os.environ)
    creds = dict((k, env[k]) for k in cred_keys if k in env)
    return creds


def zip64(string):
    """ Returns a base-64 encoded string of the zlib-compressed input string.
    The inverse operation is unzip64(), which decompresses. The zipped
    version will nearly always be smaller than the original, but for very
    short (e.g. 1 character) strings, it may be up to 11 bytes longer.
    Example:
    >>> zip64('random string')
    b'eJwrSsxLyc9VKC4pysxLBwAkcAU5\\n'
    """
    ret = binascii.b2a_base64(zlib.compress(string.encode('utf-8')))
    return ret


def unzip64(compressed_string):
    """ Given a base-64 encoded and zlib-compressed string, returns the
    original, uncompressed string. The inverse operation is zip64().
    Example:
    >>> 'some string' == unzip64(zip64('some string'))
    True"""
    ret = zlib.decompress(binascii.a2b_base64(compressed_string))
    return ret.decode('utf-8')


def sha256(string):
    """ Returns a SHA-256 value of a given string.
    Example:
    >>> sha256('hello world').__str__()
    "b'uU0nuZNNPgilLlLX2n2r+sSE7+N6U4DukIj3rOLvzek='"
    """
    return b64encode(hashlib.sha256(string.encode()).digest())


def dict2json(dictionary):
    """ Returns a JSON string given a dictionary. May not always
    return the same string because dictionaries are unordered.
    Example:
    >>> dict2json({'b': 'hello'})
    '{"b":"hello"}'
    """
    return json.dumps(dictionary, separators=(',', ':'))


def parse_iso8601(dictionary, key):
    """ Returns a dateutil object found from parsing the
    ISO8601 date string found in dictionary under key. If
    the object is not found, returns None."""
    try:
        if 'time_started' in dictionary:
            d = dateutil.parser.parse(dictionary[key])
            return d
        else:
            return None
    except:
        s = "{} not a valid ISO8601 datestring".format(key)
        raise ValueError(s)


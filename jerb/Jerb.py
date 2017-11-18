""" jerb/Jerb.py: The Jerb class: a git repo packed into an immutable JSON. """

import os
import re
import json


class Jerb ():
    def __init__(self, init_json_string):
        """Creates Jerb object from the provided JSON string. Jerbs
        are immutable, so any attempts to modify this object will cause
        errors. """

        j = json.loads(init_json_string)

        self.init_json_string = init_json_string
        self.jid = j['jid']
        self.meta = j['meta']
        self.pack = j['pack'] if 'pack' in j else None

        # Check for any internal consistency errors
        err = any(self.errors())
        if err:
            raise ValueError(err)

    def __str__(self):
        """ Returns a string that is the serialized Jerb.
        Warning: ordering of key/values is not guaranteed to be consistent """
        # TODO: Check that nobody modified this jerb.
        return self.init_json_string

    def errors(self):
        """ Returns an iterable of errors found when validating this JERB for
        self-consistency. """
        # TODO: Unpack it, and check that metadata is in the pack
        # TODO: Unpack it, and check that there is only 1 note object
        # TODO: Check that the JID hash matches the true commit
        # TODO: Check that there is a user, date, ref defined (and other req fields)
        # TODO: Check that the user, date, ref are in the correct format
        # TODO: Function that checks a date is ISO8601 compatible/strptime?
        return [None]

    def save_to_file(self, filepath):
        """ Saves this jerb to filepath, overwriting whatever was there. """
        with open(filepath, 'wb') as f:
            f.write(str(self).encode())


##############################################################################
# Helper functions since Python does not allow multiple constructors

def load_jerb_from_file(filepath):
    """ Loads a .jerb file and returns a Jerb object. """
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            init_json_string = f.read()
            s = init_json_string.decode()
            j = Jerb(s)
            return j
    else:
        raise ValueError("File not found: " + filepath)


def valid_metadata_structure(metadata):
    """ Predicate. True when metadata is the correct format: a
    dict of strings or lists of strings."""
    if type(metadata) is not dict:
        return False
    for k, v in metadata.items():
        if not k:
            return False
        if type(k) is not str:
            return False
        if v and (type(v) is not str) and (type(v) is not list):
            return False
    return True


def valid_SHA1_string(sha):
    """ Predicate. True when S is a valid SHA1 string."""
    r = re.compile('^([a-f0-9]{40})$')
    m = re.search(r, sha)
    if m:
        return True
    else:
        return False

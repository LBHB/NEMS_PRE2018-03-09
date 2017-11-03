""" JERB: JSON Executable Resource Blob Package.
See: README.org for more details.
"""

import binascii
import json


class Jerb ():
    def __init__(self, init_json_string):
        """ Inits Jerb object from the provided JSON string or filepath. """

        j = json.loads(init_json_string)

        self.jid = j['jid']
        self.meta = j['meta']
        self.pack = binascii.a2b_base64(j['pack']) if 'pack' in j else None

        # Internal storage
        self.__init_json_string__ = init_json_string

        # Internal flags: If there is no payload, we'll need to fetch it
        self.__nopayload__ = True if not (self.pack) else False

        # Check for any internal consistency errors
        err = any(self.errors())
        if err:
            raise ValueError(err)

    def as_dict(self):
        d = {'jid': self.jid,
             'meta': self.meta,
             'pack': self.pack}
        return d

    def __str__(self):
        """ Returns a string that is the serialized Jerb.
        Warning: ordering of key/values is not guaranteed to be consistent """
        s = json.dumps(self.as_dict())
        return s

    def errors(self):
        """ Returns an iterable of errors found when validating this JERB for
        self-consistency. """
        return [None]


# TODO: Check that the JID hash matches the data
#     -

# TODO: Function to compare two jerbs, and see if one is the "unexecuted"
#       version of the other "executed" JERB.  Why? Well, we'll want to know
#       that we get the same, bit-for-bit result from two different workers?

# TODO: Function to load a JID from a string

# TODO: Function to load a JID from a file

# TODO: Function to load a JID from a HTTP(s) URI

# TODO: Function to return a dictionary of properties you would like indexed

# TODO: Function to return a JSON of properties you would like indexed

# TODO: Function that adds timestamp, user information when saving a JERB

# TODO: Function that checks a date is ISO8601 compatible (strptime?)
#       parse_iso8601 may be useful

# TODO: Function that double-checks timestamps are logically correct
    # def timestamp_errors(self):
    #     """ Returns the first error found regarding timestamps. """
    #     if ((self.time_queued and self.time_started and
    #          self.time_queued > self.time_started)):
    #         return ValueError("time_started not after time_queued")

    #     if ((self.time_started and self.time_finished and
    #          self.time_started > self.time_finished)):
    #         return ValueError("time_finished not after time_started")

    #     return False

# TODO: Functions to add, get, update, or delete properties?
#       Or is it easier to just leave it as a dictionary?

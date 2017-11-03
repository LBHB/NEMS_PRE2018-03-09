""" jerbstore : An API for storing jerbs centrally
Presently, by default it stores them in a directory. """

import os
from flask import abort, request, Response
from flask_restful import Resource

from jerb.Jerb import Jerb
import jerb.lib


def ensure_valid_jid(jid):
    if not jerb.lib.is_SHA1_string(jid):
        abort(400, 'invalid SHA1 string:' + jid)


def jerb_not_found():
    # TODO: Southpark reference
    abort(404, "jerb not found")


class LocalJerbStore(Resource):
    """ A class representing a store of local jerb files."""
    def __init__(self, **kwargs):
        self.LOCAL_JERB_DIR = kwargs['local_jerb_dir']

    def jerb_path(self, jid):
        """ Returns the path for a local JID."""
        p = os.path.join(self.LOCAL_JERB_DIR, jid[0:2], jid[2:])
        return p

    def jerb_exists(self, jid):
        """ Predicate. Returns True when the JID exists locally. """
        p = self.jerb_path(jid)
        return os.path.exists(p)

    def get(self, jid):
        """ Returns the jerb found at JID, if it exists. """
        ensure_valid_jid(jid)
        if not self.jerb_exists(jid):
            jerb_not_found()
        p = self.jerb_path(jid)
        with open(p, "rb") as f:
            d = f.read()
        return Response(d, status=200, mimetype='application/json')

    def put(self, jid):
        """ Idemtpotent. Returns 201 if the jerb was created, or
        return 200 if it exists already in this jerbstore."""
        ensure_valid_jid(jid)
        # TODO: Ensure request is within limits
        # TODO: Ensure JSON is using single quotes(?)
        js = request.get_json()
        # js = json.loads(s)
        j = Jerb(js, already_json=True)
        if not jid == j.jid:
            abort(400, 'JID does not match argument')
        if any(j.errors()):
            abort(400, 'jerb contains errors')
        if not self.jerb_exists(jid):
            p = self.jerb_path(j.jid)
            d = os.path.dirname(p)
            if not os.path.isdir(d):
                os.makedirs(d)
            with open(p, 'wb') as f:
                f.write(request.data)
            return Response(status=201)
        else:
            return Response(status=200)

    def delete(self, jid):
        ensure_valid_jid(jid)
        # TODO
        jerb_not_found()

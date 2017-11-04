from flask import abort, request, Response
from flask_restful import Resource
import json

import jerb.lib
import jerb_index.redis_index as red
from jerb.Jerb import Jerb


# TODO: These two are cut and pasted. Consolidate!
def ensure_valid_jid(jid):
    if not jerb.lib.is_SHA1_string(jid):
        abort(400, 'invalid SHA1 string:' + jid)


def jerb_not_found():
    # TODO: Southpark reference
    abort(404, "jerb not found")


class JerbIndex(Resource):
    def __init__(self, **kwargs):
        self.rdb = kwargs['redisdb']

    def get(self, jid):
        """ Returns the metadata for the jerb found at JID, if it exists."""
        ensure_valid_jid(jid)
        metadata = red.lookup_jid(self.rdb, jid)
        if metadata:
            return Response(metadata, status=200, mimetype='application/json')
        else:
            return Response("Not found", status=404)

    def put(self, jid):
        """ Idempotent. Indexes the jerb."""
        ensure_valid_jid(jid)
        # TODO: Ensure request is within limits
        js = request.get_json()
        j = Jerb(js, already_json=True)
        # TODO: This is boilerplate and same as jerbserve. Consolidate.
        if not jid == j.jid:
            abort(400, 'JID does not match argument')
        if any(j.errors()):
            abort(400, 'jerb contains errors')
        red.index_jerb(self.rdb, j)
        return Response(status=200)

    def delete(self, jid):
        ensure_valid_jid(jid)
        j = red.lookup_jid(jid)
        if j:
            j = red.deindex_jerb(jid)
            return "Successfully deindexed."
        else:
            jerb_not_found()


class JerbQuery(Resource):

#    def __init__(self):
#        self.pp = reqparse.RequestParser()
#        self.pp.add_argument('prop', type=str)
#        self.pp.add_argument('val', type=str)

    def get(self, prop, val):
        return Response('Nothing to see here', 200)
    #args = self.pp.parse_args(self)
    #    red.lookup_prop(r, args['prop'], args['val'])



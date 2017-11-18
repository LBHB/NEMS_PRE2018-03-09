from flask import abort, request, Response
from flask_restful import Resource, reqparse
import json

import jerb_index.redis_index as red
from jerb.Jerb import Jerb, valid_SHA1_string, valid_metadata_structure


# TODO: These two are cut and pasted. Consolidate!
def ensure_valid_jid(jid):
    if not valid_SHA1_string(jid):
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
        j = Jerb(request.data.decode())
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


class JerbFindQuery(Resource):
    def __init__(self, **kwargs):
        self.rdb = kwargs['redisdb']

    def post(self):
        js = request.get_json()
        if not valid_metadata_structure(js):
            abort(400, 'Invalid query format')
        jids = red.select_jids_where(self.rdb, js)
        return Response(json.dumps({"jids": jids}), 200)


class JerbRefQuery(Resource):
    def __init__(self, **kwargs):
        self.rdb = kwargs['redisdb']
        self.argparser = reqparse.RequestParser()
        self.argparser.add_argument('user', type=str, help='jerb username')
        self.argparser.add_argument('ref', type=str, help='jerb ref')

    def get(self):
        args = self.argparser.parse_args()
        user = args['user']
        ref = args['ref']
        if (user and ref):
            jids = red.get_head(self.rdb, user, ref)
            return Response(json.dumps({"jids": jids}), 200)
        else:
            abort(400, 'User and ref not defined')

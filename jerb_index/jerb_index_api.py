import requests
import json
from flask import Flask
from flask_restful import Api, Resource, reqparse

import jerb
from jerb_index import redis_index

app = Flask(__name__)
api = Api(app)

creds = jerb.util.environment_credentials()
r = redis_index.redis_connect(creds)


class Docs(Resource):
    def get(self):
        return {"routes":
                {"GET /jid/*": "Returns the metadata for a given JID. ",
                 "POST /jid/*": "Indexes the Jerb payload, if not yet indexed",
                 "DELETE /jid/*": "Unindexes the Jerb payload, if indexed",
                 "GET /find?prop=_&val=_": "Returns JIDs matching prop=val"}}


class JID(Resource):
    def get(self, jid):
        # TODO: Sanitize JID
        j = redis_index.lookup_jid(jid)
        return j

    def post(self, jid):
        # TODO: Sanitize JID
        js = json.loads(requests.body.read())
        # TODO: Sanitize JS
        j = jerb.Jerb(js)
        errs = j.errors()
        if any(errs) or not (jid == j.jid):
            raise errs
        else:
            return redis_index.index_jerb(r, j)

    def delete(self, jid):
        # TODO: Sanitize JID
        j = redis_index.lookup_jid(jid)
        if j:
            j = redis_index.deindex_jerb(jid)
            return "Successfully deindexed."
        else:
            return "JID not found."


class Lookups(Resource):

    def __init__(self):
        self.pp = reqparse.RequestParser()
        self.pp.add_argument('prop', type=str)
        self.pp.add_argument('val', type=str)

    def get(self, prop, val):
        args = self.pp.parse_args(self)
        redis_index.lookup_prop(r, args['prop'], args['val'])


api.add_resource(Docs, '/')
api.add_resource(JID, '/jid', '/jid/<string:jid>')
api.add_resource(Lookups, '/find')

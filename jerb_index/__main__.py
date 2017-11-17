
from flask import Flask
from flask_restful import Api

import jerb.util
from jerb_index import redis_index
from jerb_index.jerb_index_api import JerbIndex, JerbFindQuery, JerbRefQuery

req_env_vars = ['REDIS_HOST',
                'REDIS_PORT',
                'REDIS_PASS',
                'JERB_INDEX_HOST',
                'JERB_INDEX_PORT']
creds = jerb.util.ensure_env_vars(req_env_vars)

r = redis_index.redis_connect(creds)

app = Flask(__name__)
api = Api(app)

api.add_resource(JerbIndex,
                 '/jid/<string:jid>',
                 resource_class_kwargs={'redisdb': r})

api.add_resource(JerbFindQuery,
                 '/find',
                 resource_class_kwargs={'redisdb': r})

api.add_resource(JerbRefQuery,
                 '/ref',
                 resource_class_kwargs={'redisdb': r})

app.run(port=creds['JERB_INDEX_PORT'],
        host=creds['JERB_INDEX_HOST'])

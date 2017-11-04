from flask import Flask
from flask_restful import Api

import jerb.util
from jerb_index import redis_index
from jerb_index.jerb_index_api import JerbIndex

creds = jerb.util.environment_credentials()
r = redis_index.redis_connect(creds)

app = Flask(__name__)
api = Api(app)

api.add_resource(JerbIndex,
                 '/jid',
                 '/jid/<string:jid>',
                 resource_class_kwargs={'redisdb': r})

# api.add_resource(JerbQuery, '/find')

app.run(port=3001, host='127.0.0.1')

from flask import Flask
from flask_restful import Api
from jerbstore.jerbstore_api import LocalJerbStore

app = Flask(__name__)
api = Api(app)
api.add_resource(LocalJerbStore,
                 '/jid',
                 '/jid/<string:jid>',
                 resource_class_kwargs={'local_jerb_dir': '/home/ivar/jerbs/'})

app.run(port=3000, host='127.0.0.1')

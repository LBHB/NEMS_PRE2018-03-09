from flask import Flask
from flask_restful import Api
from jerbstore.jerbstore_api import LocalJerbStore, CentralJerbStore

app = Flask(__name__)
api = Api(app)
api.add_resource(LocalJerbStore,
                 '/local/jid',
                 '/local/jid/<string:jid>',
                 resource_class_kwargs={'local_jerb_dir': '/home/ivar/jerbs/'})

api.add_resource(CentralJerbStore,
                 '/jid',
                 '/jid/<string:jid>',
                 resource_class_kwargs={'jerb_central_repo_dir':
                                        '/home/ivar/central/'})

app.run(port=3000, host='127.0.0.1')

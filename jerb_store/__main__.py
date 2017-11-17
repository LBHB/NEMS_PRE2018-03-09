from flask import Flask
from flask_restful import Api

import jerb.util
from jerb_store.jerb_store_api import LocalJerbStore, CentralJerbStore

req_env_vars = ['JERB_STORE_HOST',
                'JERB_STORE_PORT']
creds = jerb.util.ensure_env_vars(req_env_vars)

app = Flask(__name__)
api = Api(app)
api.add_resource(LocalJerbStore,
                 '/jid',
                 '/jid/<string:jid>',
                 resource_class_kwargs={'local_jerb_dir': '/home/ivar/jerbs/'})

# api.add_resource(CentralJerbStore,
#                  '/central/jid',
#                  '/central/jid/<string:jid>',
#                  resource_class_kwargs={'jerb_central_repo_dir':
#                                         '/home/ivar/central/'})

app.run(port=creds['JERB_STORE_PORT'], host=creds['JERB_STORE_HOST'])

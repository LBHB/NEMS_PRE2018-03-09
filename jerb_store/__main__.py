from flask import Flask
from flask_restful import Api

import jerb.util
import jerb_store.jerb_store_api

req_env_vars = ['JERB_STORE_HOST',
                'JERB_STORE_PORT',
                'AWS_ACCESS_KEY_ID',
                'AWS_SECRET_ACCESS_KEY']
creds = jerb.util.ensure_env_vars(req_env_vars)

app = Flask(__name__)
api = Api(app)

# Uncomment to use S3 as the jerb storage location
api.add_resource(jerb_store.jerb_store_api.S3JerbStore,
                 '/jid',
                 '/jid/<string:jid>',
                 resource_class_kwargs={'local_jerb_dir': '/home/ivar/jerbs/'})

# Uncomment to use the local filesystem as storage
# api.add_resource(jerb_store.jerb_store_api.LocalJerbStore,
#                  '/jid',
#                  '/jid/<string:jid>',
#                  resource_class_kwargs={'local_jerb_dir': '/home/ivar/jerbs/'})

# Uncomment to use a git repo as storage
# api.add_resource(jerb_store.jerb_store_api.CentralJerbStore,
#                  '/jid',
#                  '/jid/<string:jid>',
#                  resource_class_kwargs={'jerb_central_repo_dir':
#                                         '/home/ivar/central/'})

app.run(port=creds['JERB_STORE_PORT'], host=creds['JERB_STORE_HOST'])

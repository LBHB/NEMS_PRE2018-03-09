DEBUG = True

import db_connection
db = db_connection.db
SQLALCHEMY_DATABASE_URI = 'mysql+pymysql://%s:%s@%s/%s'%(db['user'],db['passwd'],db['host'],\
                                                 db['database'])
# "Enable protection against cross-site request forgery"
CSRF_ENABLED = True

# "secure key for signing data"
CSRF_SESSION_KEY = "JLbLv7XaWMsyMrjyQ1jlT9MNdeq8iNGg"

# "secure key for signing cookies"
SECRET_KEY = "JadXbQrdbfDGCEdljigC8mDmvrf9NNrI"
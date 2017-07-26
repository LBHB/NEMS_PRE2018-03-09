""" Stores config settings for database, matplotlib, flask, et cetera."""

import os
# Only used for dummy directory detection
import nems.utils as nu

# Database settings
# To specify database settings, create the nems.config.hidden subdirectory
# and store a file 'database_into.txt' there with the following format:
#   host = hosturl
#   user = username
#   passwd = password
#   database = database
# Order doesn't matter,  but each entry should be on a separate line.
db = {}
# get correct path to db info file
libmod_path = os.path.abspath(nu.__file__)
i = libmod_path.find('nems/nems')
nems_path = libmod_path[:i+10]
try:
    with open(nems_path + "config/hidden/database_info.txt","r") as f:
        for line in f:
            key,val = line.split()
            db[key] = val
    db_uri = 'mysql+pymysql://%s:%s@%s/%s'%(
                    db['user'],db['passwd'],db['host'],db['database']
                    )
except Exception as e:
    print(e)
    db_uri = 'sqlite:////path/to/database/file'
    
    
# Flask settings
class Flask_Defaults():
    """ Specifies default flask config. To override this values securely
        define a new class, in a file outside of version control, that inherits
        from this class and overwrites the desired variables.
        Then, in nems.web.nems_analysis.__init__, pass the import string
        to that file to app.config_from_object().
        
        Example (inside __init__.py):
            app.config_from_object(
                    'nems.config.hidden.Settings.New_Settings_Class'
                    )
        
        To leave the settings as the defaults, the line would instead read:
            app.config_from_object('nems.config.config.Flask_Defaults')
            
    """
    
    # Enable werkzeug debugger in-browser (*not* secure for a public server!)
    Debug = False
    SQLALCHEMY_DATABASE_URI = db_uri
    # 'Enable protection against cross-site request forgery'
    CSRF_ENABLED = True
    # 'Secure key for signing data'
    CSRF_SESSION_KEY = "super secret key"
    # 'Secure key for signing cookies'
    SECRET_KEY = "another super secret key"

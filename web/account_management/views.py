""" Reference:
    flask.pocoo.org/snippets/63/ and .../62/
    flask-wtf.readthedocs.io/en/stable/install.html
    flask-login.readthedocs.io/en/latest/
    
"""

from urllib.parse import urlparse, urljoin

from flask import jsonify, redirect, request, url_for, render_template
from flask_wtf import LoginForm
from wtforms import TextField, HiddenField
from flask_login import LoginManager, login_required, login_user, current_user

from nems_analysis import app, Session

login_manager = LoginManager()
login_manager.init_app(app)

USER_LOGIN_TABLE = 'replace this with sqlalchemy table class'

@login_manager.user_loader
def load_user(user_id):
    session = Session()
    try:
        u_id = user_id.decode('utf-8')
        user = (
                session.query(USER_LOGIN_TABLE)
                .filter(u_id == u_id)
                .all()
                )
        return user[0]
    except:
        return None
    finally:
        session.close()

class User():
    # flask-login required class and methods
    
    def __init__(self, username, password):
        self.username = username
        self.password = password
        #self.password = someKindofEncryptionFunction(password)
        self.authenticated = False
        
    def is_authenticated(self):
        return self.authenticated
    
    def is_active(self):
        return True
    
    def is_anonymous(self):
        return False
    
    def get_id(self):
        session = Session()
        user_id = (
                session.query(USER_LOGIN_TABLE.id)
                .filter(USER_LOGIN_TABLE.username == self.username)
                .first()[0]
                )
        session.close()
        return str(user_id).encode('utf-8')

@app.route('/login', methods=['GET', 'POST'])
def login_test():
    session = Session()
    _next = get_redirect_target()
    form = LoginForm()
    if form.validate_on_submit():
        user = (
                session.query(USER_LOGIN_TABLE)
                .filter(USER_LOGIN_TABLE.username == request.form['username'])
                .all()
                )
        if user():
            # login stuff goes inside here
            # should password be encrypted at this stage?
            # use flask bcrypt?
            if user.password == request.form['password']:
                user.authenticated = True
                session.add(user)
                session.commit()
                login_user(user, remember=True)
                session.close()
                
                return redirect_back('main_view')
    session.close()
    return render_template('main.html', next=_next)

@app.route('/logout')
@login_required
def logout():
    session = Session()
    user = current_user
    user.authenticated = False
    session.add(user)
    session.commit()
    session.close()
    
    return render_template('main.html')

@app.route('/register')
def register():
    USER_LOGIN_TABLE = 'replace this with sqlalchemy table class'
    session = Session()
    
    user = request.args.get('user')
    pswd = request.args.get('pswd')
    # what other fields? user group for david lab?
    
    user = USER_LOGIN_TABLE(
            user=user, pswd=pswd,
            )
    
    #session.commit()
    session.close()
    
    return jsonify(success='success message')


# Functions from snippets 62
def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and \
           ref_url.netloc == test_url.netloc

def get_redirect_target():
    for target in request.values.get('next'), request.referrer:
        if not target:
            continue
        if is_safe_url(target):
            return target

def redirect_back(endpoint, **values):
    target = request.form['next']
    if not target or not is_safe_url(target):
        target = url_for(endpoint, **values)
    return redirect(target)

           
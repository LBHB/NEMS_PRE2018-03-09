""" Reference:
    flask.pocoo.org/snippets/63/ and .../62/
    flask-wtf.readthedocs.io/en/stable/install.html
    flask-login.readthedocs.io/en/latest/
    
"""

from urllib.parse import urlparse, urljoin

import flask
from flask import redirect, request, url_for, render_template, g
from flask_login import (
        LoginManager, login_required, login_user, logout_user, current_user
        )
from flask_bcrypt import Bcrypt

from nems_analysis import app, Session, NarfUsers
from account_management.forms import LoginForm, RegistrationForm

login_manager = LoginManager()
login_manager.init_app(app)
bcrypt = Bcrypt(app)


@login_manager.user_loader
def load_user(user_id):
    session = Session()
    try:
        # get email match from user database
        # (needs to be stored as unicode per flask-login)
        sqla_user = (
                session.query(NarfUsers)
                .filter(NarfUsers.email == user_id)
                .all()[0]
                )
        # assign attrs from table object to active user instance
        user = User(
                username=sqla_user.username,
                # password should be stored as bcrypt hash
                # (generated when registered)
                password=sqla_user.password,
                group=sqla_user.group,
                sec_level=sqla_user.sec_level,
                )
        return user
    except:
        return None
    finally:
        session.close()

@app.before_request
def before_request():
    g.user = current_user
    
@app.route('/login', methods=['GET', 'POST'])
def login():
    #_next = get_redirect_target()
    form = LoginForm(request.form)
    if request.method == 'POST' and form.validate():
        user = load_user(
                user_id=str(form.email.data).encode('utf-8')
                )
        print("made it past load_user")
        if user:
            print("made it inside the 'if user:' statement")
            if bcrypt.check_password_hash(user.password, form.password.data):
                print("made it past password check")
                session = Session()
                user.authenticated = True
                session.add(user)
                session.commit()
                login_user(user, remember=True)
                session.close()
                
                flask.flash("Login successful. Welcome back %s!"%form.username)
                return redirect(url_for('main_view'))

    return render_template(
            'account_management/login.html',
            form=form, #next=_next,
            )

@app.route('/logout')
@login_required
def logout():
    user = current_user
    user.authenticated = False
    logout_user()
    
    return render_template('main.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    #_next = get_redirect_target()
    form = RegistrationForm(request.form)
    if request.method == 'POST' and form.validate():
        session = Session()
        new_user = NarfUsers(
                username = form.username.data,
                password = bcrypt.generate_password_hash(form.password.data),
                email = (form.email.data).encode('utf-8'),
                firstname = form.firstname.data,
                lastname = form.lastname.data,
                )
        
        session.add(new_user)
        session.commit()
        session.close()

        flask.flash("Registration successful - thanks %s!"%form.username)
        return redirect(url_for('main_view'))
    
    return render_template(
            'account_management/register.html',
            form=form, #next=_next
            )


class User():
    # flask-login required class and methods
    # have this inherit from sqlalchemy base? then
    # these methods just get added on, but all db fields visible as attr.
    
    def __init__(self, username, password, group=None, sec_lvl=0):
        self.username = username
        self.password = password
        self.group = group
        self.sec_level = sec_lvl
        self.authenticated = False
        
    def is_authenticated(self):
        # self.authenticated should be false until password verified in
        # login function.
        # should be reset to faulse on logout or if session ends.
        return self.authenticated
    
    def is_active(self):
        # All users active - can implement checks here (like e-mail conf)
        # if desired.
        return True
    
    def is_anonymous(self):
        # No users are anonymous - can implement this later if desired.
        return False
    
    def get_id(self):
        # user_id should be the unicode rep of e-mail address
        # (should be stored in db table in that format)
        session = Session()
        user_id = (
                session.query(NarfUsers.email)
                .filter(NarfUsers.username == self.username)
                .first()[0]
                )
        session.close()
        return user_id

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

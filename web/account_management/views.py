""" Reference:
    flask.pocoo.org/snippets/63/ and .../62/
    flask-wtf.readthedocs.io/en/stable/install.html
    flask-login.readthedocs.io/en/latest/
    
"""
    
from urlparse import urlparse, urljoin

from flask import jsonify, redirect, request, url_for, render_template
from flaskext.wtf import Form, TextField, HiddenField

from nems_analysis import app, Session


@app.route('/log_in_test')
def log_in_test():
    user = 'testing'
    success = True
    return jsonify(user=user, success=success)
    
@app.route('/log_in')
def log_in():
    USER_LOGIN_TABLE = 'replace this with sqlalchemy table class'
    session = Session()
    
    user = request.args.get('user')
    pswd = request.args.get('pswd')
    
    existing = (
            session.query(USER_LOGIN_TABLE)
            .filter(user == user)
            .first()
            )
            
    if not existing:
        # return an error message indicating no such
        # matching user/password combo
        # (same as +user/-pass for better security)
        pass
    else:
        if existing.password == pswd:
            # do stuff that logs the user in
            pass
        else:
            # return an error message indicating no matching
            # user/pass combo
            pass
            
@app.route('/log_out')
def log_out():
    return jsonify(success=True)

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

@app.route('/login', methods=['GET', 'POST'])
def login():
    _next = get_redirect_target()
    if request.method == 'POST':
        #login stuff goes inside here
        return redirect_back('index')
    return render_template('main.html', next=_next)
           
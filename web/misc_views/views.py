"""Miscellaneous view functions.

Contents so far:
    console (and supporting functions/classes)
    error_log
    
console referenced from:
    http://flask.pocoo.org/snippets/116/
    
"""
import sys

from flask import Response, request

from nems_analysis import app


@app.route('/error_log')
def error_log():
    """Serve the static error_log.txt file."""
    
    # TODO: Add an interface to edit the text from the site, or submit
    #       suggestions some other way, so that users can report bugs etc.
    return app.send_static_file('error_log.txt')


@app.route('/console')
def console():
    """Serve contents of sys.stdout as they are added."""
    print("inside console view")
    
    if request.headers.get('accept') == 'text/event-stream':
        print("inside request.headers.get('accept') if statement")
        def gen():
            print("inside gen() function")
            stdout_gen = ( line for line in sys.stdout.readlines() )
            try:
                while True:
                    
                    yield str(next(stdout_gen))
            except GeneratorExit:
                yield "Stream broken."
            
        return Response(gen(), mimetype="text/event-stream")
    print("'if request.headers.get' failed")
    return Response("Stream failed.", mimetype="text/event-stream")

    


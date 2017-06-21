"""Miscellaneous view functions.

Contents so far:
    console
    error_log
    
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


#@app.route('/py_console')
#def console():
#    """Serve contents of sys.stdout as they are added."""

    #stdout_gen = ( line for line in sys.stdout )
#    def gen():
#        while True:
#            yield 'test message'
#            
#    return Response(gen(), mimetype="text/event-stream")

    


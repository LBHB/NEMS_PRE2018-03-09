"""Miscellaneous view functions.

Contents so far:
    error_log
    
"""

from flask import Response, request

from nems_analysis import app


@app.route('/error_log')
def error_log():
    """Serve the static error_log.txt file."""
    
    # TODO: Add an interface to edit the text from the site, or submit
    #       suggestions some other way, so that users can report bugs etc.
    return app.send_static_file('error_log.txt')




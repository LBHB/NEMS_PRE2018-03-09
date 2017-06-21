"""Miscellaneous view functions.

Contents so far:
    console
    error_log
    
"""
import sys

from flask import Response, request

from nems_analysis import app, socketio, thread


@app.route('/error_log')
def error_log():
    """Serve the static error_log.txt file."""
    
    # TODO: Add an interface to edit the text from the site, or submit
    #       suggestions some other way, so that users can report bugs etc.
    return app.send_static_file('error_log.txt')

# event source
#@app.route('/py_console')
#def console():
#    """Serve contents of sys.stdout as they are added."""

    #stdout_gen = ( line for line in sys.stdout )
#    def gen():
#        while True:
#            yield 'test message'
            
#    return Response(gen(), mimetype="text/event-stream")

# socketio

def py_console():
    count = 0
    while True:
        socketio.sleep(5)
        print('inside py_console while loop')
        count += 1
        socketio.emit(
                'console_update',
                {'data':'%d'%count}, 
                namespace='/py_console',
                )

@socketio.on('connect', namespace='/py_console')
def start_logging():
    global thread
    if thread is None:
        print('adding background function to thread')
        thread = socketio.start_background_task(target=py_console)
    socketio.emit(
            'console_update',
            {'data':'connected'},
            namespace='/py_console',
            )
    
@socketio.on('jsconnect', namespace='/py_console')
def print_js_emit(data):
    print('jsconnect event received: ' + str(data))
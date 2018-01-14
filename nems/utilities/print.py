"""DO NOT USE FOR THE TIME BEING, might be re-implemented at a later date.
    -jacob 1/13/2018
"""

def web_print(s):
    print(s)
    return

"""
    use_app = False
    try:
        from nems_web.nems_analysis import app
        from nems_web.nems_analysis import socketio
        use_app = True
    except Exception as e:
        print('Error when using web_print')
        print(e)
        socketio = None
        use_app = False
    if use_app:
        s = s.replace('\n', '<br>')
        socketio.emit(
            'console_update',
            {'data': s},
            namespace='/py_console',
        )
    else:
        print(s)
"""
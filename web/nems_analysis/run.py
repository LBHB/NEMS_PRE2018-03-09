"""Module for launching the nems_analysis flask app.

Launches the app and hosts the server at the domain/ip and port
specified. See flask documentation for additional app.run options.

Note:
-----
debug should NEVER be set to True when launching the server for a
'production' environment (i.e. when hosting for public use). Per the
flask documentation, this would allow users to run "arbitrary code" on
the server, potentially causing harm.

"""

from nems_analysis import app

app.run(host='0.0.0.0',port=8000,debug=True)
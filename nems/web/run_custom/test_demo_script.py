""" test code for demo_script.py

should return an html string that, if copy and pasted into an html doc with
bokeh's dependencies linked, will display a bokeh plot.

the plot should also open on its own in a web browser when the function is run.

"""

import nems.web.run_custom.demo_script as demo
# This information would normally be passed by the web interface,
# but it can also be specified manually.
argsdict = {
        'batch' : 291,
        'models' : ['fb18ch100_wc01_fir15_dexp_fititer00',
                    'fb18ch100_wc01_stp1pc_fir15_dexp_fititer00'
                    ],
        'cells' : ['bbl031f-a1', 'bbl034e-a1', 'bbl034e-a2', 'bbl036e-a1',
                   'bbl036e-a2', 'bbl038f-a1', 'bbl038f-a2'
                   ],
        'onlyFair' : True,
        'includeOutliers' : False,
        'snr' : 0.0,
        'iso' : 75,
        'snri' : 0.1,
        'measure' : 'r_test',
        }

html = demo.run_script(argsdict)
print(html)
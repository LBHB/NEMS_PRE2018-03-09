"""
- Plot Generator object
- Sets attributes based on variables passed from views.py
- Then obtains data from Query Generator to be used in plots
- Makes plot using ?bokeh or matplotlib? then returns it in a 
- template-friendly format (or just pops it up in a new window if bokeh)
"""
#TODO: everything.
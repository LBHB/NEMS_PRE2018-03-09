"""
- Plot Generator object
- Sets attributes based on variables passed from views.py
- Then obtains data from Query Generator to be used in plots
- Makes plot using ?bokeh or matplotlib? then returns it in a 
- template-friendly format (or just pops it up in a new window if bokeh)
"""

from bokeh.plotting import *
from bokeh.resources import CDN
from bokeh.embed import file_html, components
import pandas as pd
import pandas.io.sql as psql
#used for checking empty query
import webbrowser

class PlotGenerator():
    # currently just generates a scatter plot for model performance
    # TODO: add functionality for additional plot types
    def __init__(self, connection, plottype='Scatter', tablename='NarfResults',\
                 batchnum = '', modelnameX = '', modelnameY = '', measure = ''):
        # pass through db connection for communication with
        # QueryGenerator
        
        self.connection = connection
        self.plottype = plottype
        self.tablename = tablename
        self.batchnum = batchnum
        # one model for x axis, other for y axis
        """modelnames for plot testing 
        self.modelnameX = "fb18ch100_lognn_wcg03_ap3z1_dexp_fit05v"
        self.modelnameY = "fb18ch100_lognn_wcg03_voltp_ap3z1_dexp_fit05v" 
            modelnames for plot testing"""
        self.modelnameX = modelnameX
        self.modelnameY = modelnameY
        # measure of performance, i.e. r_test, r_ceiling etc
        self.measure = measure
        
        """ extra attributes for debugging"""
        self.dataX = []
        self.dataY = []
        """  """
        
        self.plot = self.get_plot()
        
    def get_plot(self):
        # query database filtered by batch id and where modelname must be one of
        # either model x or model y
        
        data = psql.read_sql('SELECT * FROM %s WHERE (batch="%s") AND ((modelname="%s")\
                                                      OR (modelname="%s"))'\
                             %(self.tablename,self.batchnum, self.modelnameX,\
                             self.modelnameY), self.connection)
    
        """
        # temporary hardcoded query for testing plot creation
        data = psql.read_sql('SELECT * FROM NarfResults WHERE (batch=271) AND \
                             ((modelname="fb18ch100_lognn_wcg03_ap3z1_dexp_fit05v") OR \
                             (modelname="fb18ch100_lognn_wcg03_voltp_ap3z1_dexp_fit05v"))', \
                              self.connection)
        """
        
        if data.size == 0:
            webbrowser.open_new_tab("0.0.0.0:8000/empty")
            
        clist = list(data['cellid'])
        
        xvalues = []
        yvalues = []
        
        # mask dataframe by expression result then pull out series
        modelX = data[(data.modelname == self.modelnameX)]
        modelY = data[(data.modelname == self.modelnameY)]      
        
        for cell in clist:
            #if cell id is present in both the modelX and modelY series
            if (cell in modelX.cellid.values.tolist()) and \
            (cell in modelY.cellid.values.tolist()):
                # then both models were applied to the cell, so it is valid for plot
                # apply mask to isolate unique cell + model combination (should only be one)
                xmeas = data[(data.modelname.str.match(self.modelnameX)) & (data.cellid == cell)]
                # get the value of the performance measure and add to list of x values
                xvalues.append(data.get_value(xmeas.index.values[0],'%s'%self.measure))
                # then repeat for y values
                ymeas = data[(data.modelname.str.match(self.modelnameY)) & (data.cellid == cell)]
                yvalues.append(data.get_value(ymeas.index.values[0],'%s'%self.measure))
               
        """ debug attrib """
        self.dataX.extend(xvalues)
        self.dataY.extend(yvalues)
        """ debug attrib """
        
        p = figure(x_range=[0,1],y_range=[0,1])
        p.circle(xvalues, yvalues, size=2, color='navy', alpha=0.5)
        p.line([0,1],[0,1], line_width=1, color='black')
        
        # p.show() this pops up a plot in a new tab, but only server-side
        # would need to embed bokeh server via iframe or auto_server load
        
        # returns the plot as html file..
        #return file_html(p, CDN, "Scatter Plot")
        
        # returns script and div, in that order, in a tuple
        # embed in html to display plot script.
        # still can't figure out how to open as new tab, but should be able
        # to embed on existing page eventually with ajax implementation
        # also allows extra templating of resulting plot page to add other links etc
        script,div = components(p)
        return (script,div)
        
    
        
        
        
        
"""
- Plot Generator objects
- Sets attributes based on variables passed from views.py
- Makes plot using bokeh then returns it in a 
- template-friendly format
"""

from bokeh.plotting import figure
from bokeh.io import gridplot
from bokeh.embed import components
import pandas as pd
import itertools

class PlotGenerator():
    def __init__(self, data = 'dataframe', celllist = '', modelnames = '',\
                 measure = 'r_test'):
        # list of models of interest
        self.data = data
        self.modelnames = modelnames
        self.celllist = celllist
        # measure of performance, i.e. r_test, r_ceiling etc
        self.measure = measure

    # all plot classes should implement this method to return a script and a div
    def generate_plot(self):
        self.script, self.div = ('','')

class Scatter_Plot(PlotGenerator):
        # query database filtered by batch id and where modelname must be one of
        # either model x or model y
    def __init__(self, data = 'dataframe', celllist = '', modelnames = '',\
                 measure = 'r_test'):
        PlotGenerator.__init__(self,data,celllist,modelnames,measure)
        
        
    def generate_plot(self):
        
        # keep a list of the plots generated for each model combination
        plots = []
        
        # returns a list of tuples representing all pairs of models
        for pair in list(itertools.combinations(self.modelnames,2)):
            xvalues = []
            yvalues = []
            
            modelX = self.data[(self.data.modelname == pair[0])]
            modelY = self.data[(self.data.modelname == pair[1])]
            
            for cell in self.celllist:
                # only plot cell if it has a value for both models
                if (cell in modelX.cellid.values.tolist()) and \
                   (cell in modelY.cellid.values.tolist()):
                       #find unique cell + model combo in dataframe
                       xmeas = self.data[(self.data.modelname.str.match(pair[0])) &\
                               (self.data.cellid == cell)]
                       #get value of measure for that combo and append it to values
                       xvalues.append(self.data.get_value(xmeas.index.values[0],self.measure))
                       
                       #then do the same for y-axis model
                       ymeas = self.data[(self.data.modelname.str.match(pair[1])) &\
                               (self.data.cellid == cell)]
                       yvalues.append(self.data.get_value(ymeas.index.values[0],self.measure))
            
            if (len(xvalues) > 0) and (len(yvalues) > 0):
                p = figure(x_range = [0,1], y_range = [0,1],x_axis_label=pair[0],\
                           y_axis_label=pair[1])
            
                p.circle(xvalues,yvalues,size=2,color='navy',alpha=0.5)
                p.line([0,1],[0,1],line_width=1,color='black')
            
                plots.append(p)
        
        # if made more than one plot (i.e. more than 2 models selected),
        # put them in a grid
        if len(plots) > 1:
            # split plot list into a list of lists based on size compared to
            # nearest perfect square
            i = 1
            while i**i < len(plots):
                i += 1
                
            nestedplots = [plots[j:j+i] for j in range(0,len(plots),i)]
            
            grid = gridplot(nestedplots)
            
            self.script,self.div = components(grid)
        # otherwise just return the one plot
        else:
            p = plots[0]
            self.script,self.div = components(p)
            

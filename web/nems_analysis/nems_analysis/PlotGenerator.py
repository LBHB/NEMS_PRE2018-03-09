"""
- Plot Generator objects
- Sets attributes based on variables passed from views.py
- Makes plot using bokeh then returns it in a 
- template-friendly format
"""

from bokeh.plotting import figure
from bokeh.io import gridplot
from bokeh.embed import components
from bokeh.models import ColumnDataSource,HoverTool,ResizeTool,SaveTool,\
                            WheelZoomTool,PanTool,ResetTool
import pandas as pd
import itertools

class PlotGenerator():
    def __init__(self, data = 'dataframe', celllist = '', modelnames = '',\
                 measure = 'r_test', batch=''):
        # list of models of interest
        self.data = data
        self.modelnames = modelnames
        self.celllist = celllist
        # measure of performance, i.e. r_test, r_ceiling etc
        self.measure = measure

    # all plot classes should implement this method to return a script and a div
    # to be used by plot.html template
    def generate_plot(self):
        self.script, self.div = ('','')
    
    # plot classes should implement this method to include a hover tooltip
    # must take plot data frame a column data source to use this
    def create_hover(self):
        hover_html = 'generated html code'
        return HoverTool(tooltips=hover_html)
        # example code snippet from fullstackpython.com
        #    hover_html = """
        #        <div>
        #            <span class="hover-tooltip">$x</span>
        #        </div>
        #        ..repeat
        #   return HoverTool(tooltips=hover_html)
        
        #   "$x displays the plot's xaxis. @x would show the 
        #   'x' field from the data source"
        
        """ example usage within generate_plot"""
        """
            hover_tool = self.create_hover()
            plot = figure(title=x, x_range=1.... tools=[hover_tool,other_tool])
        """
        
class Scatter_Plot(PlotGenerator):
        # query database filtered by batch id and where modelname must be one of
        # either model x or model y
    def __init__(self, data = 'dataframe', celllist = '', modelnames = '',\
                 measure = 'r_test', batch=''):
        PlotGenerator.__init__(self,data,celllist,modelnames,measure,batch)
        
    def create_hover(self):
        hover_html = """
            <div>
                <span class="hover-tooltip">x: $x</span>
            </div>
            <div>
                <span class="hover-tooltip">y: $y</span>
            </div>
            <div>
                <span class="hover-tooltip">cell: @cellid</span>
            </div>
            """
        return HoverTool(tooltips=hover_html)
            
    def generate_plot(self):
        # keep a list of the plots generated for each model combination
        tools = [PanTool(),ResizeTool(),WheelZoomTool(),SaveTool(),ResetTool(),\
                 self.create_hover()]
        plots = []
        
        # TODO: Re-do this to generate plots from a column data source
        #       so that additional tools can be used (like a hover tooltip)
        #       Plan: put xvalues, yvalues, and any desired columns into a new
        #       dataframe to be stored in a list, so that plots[i] and frames[i]
        #       point to corresponding plot + data source.
        
        # returns a list of tuples representing all pairs of models
        for pair in list(itertools.combinations(self.modelnames,2)):
            # split dataframe into rows that match each model
            modelX = self.data.loc[(self.data['modelname'] == pair[0])]
            modelY = self.data.loc[(self.data['modelname'] == pair[1])]

            # create empty dataframe 
            dat_source = pd.DataFrame()
            
            for cell in self.celllist:
                # if cell is in dataframe for both models
                if (cell in modelX.cellid.values.tolist()) and \
                   (cell in modelY.cellid.values.tolist()):
                       x = modelX.loc[modelX['cellid'] == cell]
                       y = modelY.loc[modelY['cellid'] == cell]
                       # append appropriate row from each dataframe
                       dat_source = dat_source.append([x,y],ignore_index=True)
                       
            if dat_source.size > 0:
                # split dataframe on every other row so that modelname and measure
                # are grouped into two columns each corresponding to x and y values
                dat_source = pd.DataFrame({'x':dat_source[self.measure].iloc[::2]\
                                           .values,\
                                           'y':dat_source[self.measure].iloc[1::2]\
                                           .values,\
                                           'cellid':dat_source['cellid'].iloc[::2]\
                                           .values,\
                                           'modelname_x':dat_source['modelname']\
                                           .iloc[::2].values,\
                                           'modelname_y':dat_source['modelname']\
                                           .iloc[1::2].values})

                dat_source = ColumnDataSource(dat_source)
                
                p = figure(x_range = [0,1], y_range = [0,1],x_axis_label=pair[0],\
                           y_axis_label=pair[1], title=self.measure, tools=tools)
            
                p.circle('x','y',size=5,color='navy',alpha=0.5,source=dat_source)
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
                
            # plots may still wrap around at a specific limit regardless, depending
            # on size and browser
            nestedplots = [plots[j:j+i] for j in range(0,len(plots),i)]
            grid = gridplot(nestedplots)
            
            self.script,self.div = components(grid)
        # otherwise just return the one plot
        elif len(plots) == 1:
            p = plots[0]
            self.script,self.div = components(p)
        else:
            self.script, self.div = ('Error,','No plots to display.')

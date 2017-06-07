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
                            WheelZoomTool,PanTool,ResetTool,Range1d,FactorRange
from bokeh.charts import Bar, BoxPlot
from bokeh.models.glyphs import VBar
import pandas as pd
import numpy as np
import itertools


"""
setting default tools as global variable was causing issues with scatter
plot, included here as comment for copy-paste as needed instead.

tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     self.create_hover()]

"""


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

        
class Scatter_Plot(PlotGenerator):
    
    def __init__(self, data = 'dataframe', celllist = '', modelnames = '',\
                 measure = 'r_test', batch=''):
        PlotGenerator.__init__(self,data,celllist,modelnames,measure,batch)
        
    def create_hover(self):
        hover_html = """
            <div>
                <span class="hover-tooltip">%s x: $x</span>
            </div>
            <div>
                <span class="hover-tooltip">%s y: $y</span>
            </div>
            <div>
                <span class="hover-tooltip">cell: @cellid</span>
            </div>
            """%(self.measure,self.measure)
        return HoverTool(tooltips=hover_html)
            
    def generate_plot(self):
        # keep a list of the plots generated for each model combination
        plots = []

        # returns a list of tuples representing all pairs of models
        for pair in list(itertools.combinations(self.modelnames,2)):
            tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     self.create_hover()]
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
                       # in x --> y order
                       # should only be one x row and one y row per cell
                       # since self.data was filtered by batch
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
        if len(plots) >= 1:
            # split plot list into a list of lists based on size compared to
            # nearest perfect square
            i = 1
            while i**i < len(plots):
                i += 1
                
            # plots may still wrap around at a specific limit regardless, depending
            # on size and browser
            if i == 1:
                singleplot = plots[0]
                self.script,self.div = components(singleplot)
                return
            else:
                nestedplots = [plots[j:j+i] for j in range(0,len(plots),i)]
                
            grid = gridplot(nestedplots)

            self.script,self.div = components(grid)

        else:
            self.script, self.div = ('Error,','No plots to display.')


class Bar_Plot(PlotGenerator):
    
        def __init__(self, data = 'dataframe', celllist = '', modelnames = '',\
                 measure = 'r_test', batch=''):
            PlotGenerator.__init__(self,data,celllist,modelnames,measure,batch)
            
        def create_hover(self):
            hover_html = """
                <div>
                    <span class="hover-tooltip">model: $x</span>
                </div>
                <div>
                    <span class="hover-tooltip">mean: @mean</span>
                </div>
                <div>
                    <span class="hover-tooltip">stdev: @stdev</span>
                </div>
            """
            return HoverTool(tooltips=hover_html)
            
        def generate_plot(self):
            
            # TODO: build custom bar chart instead of using bokeh's built-in
            #       chart, since it doesn't support a data source for use with
            #       hover tooltip.
            # OR:   render stdev information some other way
            # NOTE: """ commented areas are for using column data source,
            #       still has bugs to work out. need to figure out how to
            #       space categories along x axis instead of clumping them all
            #       in one spot (i.e. specify 'tick' spacing)
            
            #build new pandas series of stdev values to be added to dataframe

            stdev_col = pd.Series(index=self.data.index)
            mean_col = pd.Series(index=self.data.index)
            
            #for each model, find the stdev over the measure values, then
            #assign that value at every index that matches its modelname
            for model in self.modelnames:
                values = self.data.loc[self.data['modelname'] == model]\
                                       [self.measure].values
                stdev = np.std(values,axis=0)
                #=mean = np.mean(values,axis=0)
                indices = self.data.loc[self.data['modelname'] == model]\
                                        .index.tolist()
                for i in indices:
                    stdev_col.iat[i] = stdev
                    #=mean_col.iat[i] = mean
                    
            self.data = self.data.assign(stdev=stdev_col,mean=mean_col)
            
            tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     HoverTool()]
            
            """
            #build data source from dataframe for use with hover tool
            dat_source = ColumnDataSource(self.data)
            
            
            
            xrange = FactorRange(factors=self.data['modelname'].tolist())
            yrange = Range1d(start=0,end=max(self.data[self.measure])*1.5)
            
            p = figure(x_range=xrange,x_axis_label='Model',y_range=yrange, y_axis_label=\
                       'Mean %s'%self.measure,title="Mean %s Performance By Model"\
                       %self.measure,tools=tools)
            
            p.xaxis.major_label_orientation=(np.pi/4)
            
            glyph = VBar(x='modelname',top=self.measure,bottom=0,width=1,fill_color='navy')
            
            p.add_glyph(dat_source,glyph)
            """
            
            
            p = Bar(self.data,label='modelname',values=self.measure,agg='mean',\
                    title='Mean %s Performance By Model'%self.measure,legend=None,\
                    tools=tools, color='modelname')
            
            self.script,self.div = components(p)
            
            
            
class Pareto_Plot(PlotGenerator):
            
    def __init__(self, data = 'dataframe', celllist = '', modelnames = '',\
                 measure = 'r_test', batch=''):
        PlotGenerator.__init__(self,data,celllist,modelnames,measure,batch)
            
    def create_hover(self):
        hover_html = """
        <div>
            <span class="hover-tooltip">parameters: $x</span>
        </div>
        <div>
            <span class="hover-tooltip">mean: $y</span>
        </div>
        """
        return HoverTool(tooltips=hover_html)
            
    def generate_plot(self):
            
        tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     self.create_hover()]
            
        
        # TODO: Change this to custom char type? Not quite the same as narf pareto
        #       Currently displays bokeh default box plot with built-in
        #       summary statistics:
        #       -'whiskers' cover range of values outside of the 0.25 and 0.75
        #           quartile marks
        #       -edges of boxes represent 0.25 and 0.75 quartiles
        #       -line within box represents mean value
        #       -markers outside of whiskers represent outlier values
        
        p = BoxPlot(self.data,values=self.measure,label='n_parms',\
                        title="Mean Performance (%s) versus Complexity"%self.measure,\
                        tools=tools, color='n_parms')
            
        self.script,self.div = components(p)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
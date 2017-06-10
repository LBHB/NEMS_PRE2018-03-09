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
class DataPoint():
    # blank slate python object for storing measure variables
    def __init__(self):
        pass

class PlotGenerator():
    def __init__(self, data = 'dataframe', fair = True, outliers = False,\
                 measure = 'r_test'):
        # list of models of interest
        self.data = self.form_data_array(data)
        self.fair = fair
        self.outliers = outliers
        #put a list around measure for now so that data array can be coded
        #to accept more than one measure if desired
        self.measure = [measure]

    # all plot classes should implement this method to return a script and a div
    # to be used by plot.html template
    def generate_plot(self):
        self.script, self.div = ('','')
    
    # plot classes should implement this method to include a hover tooltip
    # must take plot data frame a column data source to use this
    def create_hover(self):
        hover_html = 'generated html code'
        return HoverTool(tooltips=hover_html)
    
    def form_data_array(self, data):
        # See: narf_analysis --> compute_data_matrix
        
        #TODO: need to set up a view to test this with some actual data
        if data.size == 0:
            return data
        
        celllist = [cell for cell in data['cellid'].values.tolist()]
        modellist = [model for model in data['modelname'].values.tolist()]
        
        # re-form NarfResults entries into dataframe of DataPoint objects,
        # one for each cellid and modelname combination with an attribute for
        # each measure (right now always 1, but could be more)
        
        # create a new dataframe of NaN values with size = to #cellids x #models
        newData = pd.DataFrame(np.nan,index=celllist,columns=modellist,dtype=object)
        
        for c in celllist:
            for m in modellist:
                dat = DataPoint()
                
                for meas in self.measure:
                    
                    if not self.outliers:
                        #if outliers is false, run a bunch of checks based on
                        # measure and if a check fails, step out of loop
                        break
                    
                    try:
                        value = data[(data.modelname == m) & (data.cellid == c)]\
                            [meas].values.tolist()[0]
                        setattr(dat,self.measure,value)
                    except IndexError:
                        # index error means values.tolist() returned list w/ 0 elements,
                        # so no measure was not recorded for this cell+model combo
                        break
                
                # if number of attributes of dat is 0, break
                # otherwise, position c,m in dataframe assigned to dat
                if len([attr for attr in dat.__dict__.iteritems()]) == 0:
                    break
                else:
                    newData.at[c,m] = dat
            
            if self.fair:
                # if fair is true, drop all rows that contain at least one np.nan value
                newData.dropna(inplace=True)
        
        return newData
    
        
        
class Scatter_Plot(PlotGenerator):
    
    def __init__(self, data = 'dataframe', fair = True, outliers = False,\
                 measure = 'r_test'):
        PlotGenerator.__init__(self,data,fair,outliers,measure)
        
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
        if self.data.size == 0:
            self.script, self.div = ('empty','plot')
            return
            
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
    
        def __init__(self, data = 'dataframe', fair = True, outliers = False,\
                 measure = 'r_test'):
            PlotGenerator.__init__(self,data,fair,outliers,measure)
            
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
            
            # TODO: add significance information (see plot_bar_pretty and randttest
                                                  # in narf_analysis)
            
            if self.data.size == 0:
                self.script, self.div = ('empty','plot')
            return
    
            #build new pandas series of stdev values to be added to dataframe
            
            #if want to show more info on tooltip in the future, just need
            #to build an appropriate series to add and then build its tooltip
            #in the create_hover function
            
            index = range(len(self.modelnames))
            stdev_col = pd.Series(index=index)
            mean_col = pd.Series(index=index)
            model_col = pd.Series(index=index,dtype=str)
            
            #for each model, find the stdev and mean over the measure values, then
            #assign those values to new Series objects to use for the plot
            i = 0
            for model in self.modelnames:
                values = self.data.loc[self.data['modelname'] == model]\
                                       [self.measure].values
                                       
                stdev = np.std(values,axis=0)
                mean = np.mean(values,axis=0)
                
                stdev_col.iat[i] = stdev
                mean_col.iat[i] = mean
                model_col.iat[i] = model
                i += 1
                    
            newData = pd.DataFrame.from_dict({'stdev':stdev_col,'mean':mean_col,\
                                              'modelname':model_col})
            
            tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     self.create_hover()]
            
            #build data source from new dataframe for use with hover tool
            dat_source = ColumnDataSource(newData)
            
            xrange = FactorRange(factors=newData['modelname'].tolist())
            yrange = Range1d(start=0,end=max(newData['mean'])*1.5)
            
            p = figure(x_range=xrange,x_axis_label='Model',y_range=yrange, y_axis_label=\
                       'Mean %s'%self.measure,title="Mean %s Performance By Model"\
                       %self.measure,tools=tools)
            
            p.xaxis.major_label_orientation=(np.pi/4)
            
            glyph = VBar(x='modelname',top='mean',bottom=0,width=0.5,fill_color='#FF5740',\
                         line_color='modelname')
            
            p.add_glyph(dat_source,glyph)

            
            #use this for basic bar plot instead, but doesn't work well with custom
            #hover tool
            """            
            p = Bar(self.data,label='modelname',values=self.measure,agg='mean',\
                    title='Mean %s Performance By Model'%self.measure,legend=None,\
                    tools=tools, color='modelname')
            """
            
            self.script,self.div = components(p)
            
            
            
class Pareto_Plot(PlotGenerator):
            
    def __init__(self, data = 'dataframe', fair = True, outliers = False,\
                 measure = 'r_test'):
        PlotGenerator.__init__(self,data,fair,outliers,measure)
            
    def create_hover(self):
        hover_html = """
        <div>
            <span class="hover-tooltip">parameters: $x</span>
        </div>
        <div>
            <span class="hover-tooltip">%s value: $y</span>
        </div>
        """%self.measure
        return HoverTool(tooltips=hover_html)
            
    def generate_plot(self):
            
        if self.data.size == 0:
            self.script, self.div = ('empty','plot')
            return
        
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
        
        #       narf version plots line covering mean +/- stdev
        #       could implement simlar to custom bar above and use either lines
        #       or narrow rectangles (for better visibility)
        p = BoxPlot(self.data,values=self.measure,label='n_parms',\
                        title="Mean Performance (%s) versus Complexity"%self.measure,\
                        tools=tools, color='n_parms')
            
        self.script,self.div = components(p)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
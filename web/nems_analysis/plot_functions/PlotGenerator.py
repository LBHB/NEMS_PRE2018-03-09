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
from bokeh.models.glyphs import VBar,Circle
import pandas as pd
import numpy as np
import math
import itertools


"""
setting default tools as global variable was causing issues with scatter
plot, included here as comment for copy-paste as needed instead.

tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     self.create_hover()]

"""

class PlotGenerator():
    def __init__(self,data='dataframe',fair=True,outliers=False,measure='r_test',\
                 extra_cols=[]):
        # list of models of interest
        self.measure = [measure]
        self.fair = fair
        self.outliers = outliers
        self.extra_cols = extra_cols
        self.data = self.form_data_array(data)
        
        # use this inside views function to check whether generate_plot
        # should be invoked
        if (self.data.size == 0):
            self.emptycheck = True
        else:
            self.emptycheck = False
            
        #put a list around measure for now so that data array can be coded
        #to accept more than one measure if desired


    # all plot classes should implement this method to return a script and a div
    # to be used by plot.html template
    def generate_plot(self):
        self.script, self.div = ('','')
    
    # plot classes should implement this method to include a hover tooltip
    # must take plot data frame a column data source to use this
    def create_hover(self):
        #if class doesn't specify code for create_hover, will just return
        #the default bokeh hover tool
        return HoverTool()
    
    def form_data_array(self, data):
        # See: narf_analysis --> compute_data_matrix
        
        # takes dataframe passed by views function (should be a full NarfResults query)
        # and converts to new multi-indexed frame (cellid level 0, modelname level 1)
        # with a column for each performance measure (base class), plus any other 
        # information needed for specific plots (overwritten in that class)
        # if 'outliers' isn't checked, filters out rows that don't meet criteria
        # if 'fair' is checked, filters out rows that contain NaN for any measure
        
        celllist = [cell for cell in list(set(data['cellid'].values.tolist()))]
        modellist = [model for model in list(set(data['modelname'].values.tolist()))]
        
        # use lists of unique cell and model names to form a multiindex for dataframe
        multiIndex = pd.MultiIndex.from_product([celllist,modellist],names=['cellid','modelname'])
        newData = pd.DataFrame(index=multiIndex,columns=self.measure+self.extra_cols)
        newData.sort_index()
        # create a new dataframe of empty values with multi index and a column
        # for each measure type
        
        for c in celllist:
            for m in modellist:
                
                dataRow = data.loc[(data.cellid == c) & (data.modelname == m)]
                
                # also add column values for any additional columns specificed
                # in plot class (ex: n_parms for pareto plot)
                if len(self.extra_cols) > 0:
                    for col in self.extra_cols:
                        try:
                            colval = dataRow[col].values.tolist()[0]
                        except:
                            # TODO: is this a good way to do this? This requires that
                            # all extra columns also have values for every cell/model
                            # if fair is checked.
                            colval = math.nan
                            
                        newData[col].loc[c,m] = colval
                
                for meas in self.measure:
                    value = math.nan
                    newData[meas].loc[c,m] = value
                    # if hit continue, value will be left as nan
                    # otherwise, will be assigned value from data after checks
                    
                    try:
                        # if measure recorded for c/m combo, assign to value
                        value = dataRow[meas].values.tolist()[0]
                    except:
                        # otherwise, error means no value recorded so leave as NaN
                        # no need to run outlier checks if value missing
                        continue
                    
                    if not self.outliers:
                        #if outliers is false, run a bunch of checks based on
                        #measure and if a check fails, step out of loop
                        
                        # commented labels from narf_analysis version
                        # "drop r_test values below threshold"
                        if ((meas == 'r_test') and (value\
                             < dataRow['r_floor'].values.tolist()[0])) or\
                           ((meas == 'r_ceiling') and (dataRow['r_test'].values.tolist()[0]\
                             < dataRow['r_floor'].values.tolist()[0])) or\
                           ((meas == 'r_active') and (value\
                             < dataRow['r_floor'].values.tolist()[0])):
                               continue
                    
                        # "drop MI values greater than 1"
                        if ((meas == 'mi_test') and (value > 1)) or\
                           ((meas == 'mi_fit') and ((value < 0) or (value > 1))):
                               continue
                           
                        # "drop MSE values greater than 1.1"
                        if ((meas == 'mse_test') and (value > 1.1)) or\
                           ((meas == 'mse_fit') and (value > 1.1)):
                               continue
                           
                        # "drop NLOGL outside normalized region"
                        if ((meas == 'nlogl_test') and ((value < -1.0) or value > 0)) or\
                           ((meas == 'nlogl_fit') and ((value < -1.0) or value > 0)):
                               continue
                           
                        # TODO: is this still used? not listed in NarfResults
                        # "drop gamma values that are too low"
                        if ((meas == 'gamma_test') and (value < 0.15)) or\
                           ((meas == 'gamma_fit') and (value < 0.15)):
                               continue
                            
                        # TODO: is an outlier check needed for cohere_test
                        #       and/or cohere_fit?
                        
                    # if value existed and passed outlier checks
                    # re-assign it to dataframe position to overwrite nan
                    newData[meas].loc[c,m] = value

        if self.fair:
            # if fair is true, drop all rows that contain a NaN value for any
            # measure column
            for c in celllist:
                for m in modellist:
                    # if any of the column values for c,m combo is null
                    if newData.loc[c,m].isnull().values.any():
                        # drop all values for all cell/model matches for this cell
                        newData.drop(c,level='cellid',inplace=True)
                        # then break out of model loop to continue to next cell
                        break
        
        # switch levels so that modelname is now primary indexer,
        # since most plots group by model
        newData = newData.swaplevel(axis=0)

        #leaving these in for testing to make sure dropping nan values
        # is working correctly
        print("was fair checked?")
        print(self.fair)
        print("does the data look different or contain nans?")
        print(newData[self.measure[0]].values)
        
        #print("printing converted data array")
        #print(newData)
        
        #print("testing index slice, should result in list of unique cells")
        #print(newData.loc['fb18ch100_lognn_wc02_fir15_siglog100_fit05h_fit05c'])
        
        return newData
    
        
        
class Scatter_Plot(PlotGenerator):
    
    def __init__(self,data='dataframe',fair=True,outliers=False,measure='r_test'):
        PlotGenerator.__init__(self,data,fair,outliers,measure)
        
    def create_hover(self):
        hover_html = """
            <div>
                <span class="hover-tooltip">%s x: @x_values</span>
            </div>
            <div>
                <span class="hover-tooltip">%s y: @y_values</span>
            </div>
            <div>
                <span class="hover-tooltip">cell: @cellid</span>
            </div>
            """%(self.measure[0],self.measure[0])
        return HoverTool(tooltips=hover_html)
            
    def generate_plot(self):
        # keep a list of the plots generated for each model combination
        plots = []
        modelnames = self.data.index.levels[0].tolist()
        
        # returns a list of tuples representing all pairs of models
        for pair in list(itertools.combinations(modelnames,2)):
            tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     self.create_hover()]
            # modelnames for x and y axes
            modelX = pair[0]
            modelY = pair[1]
            
            dataX = self.data.loc[modelX]
            dataY = self.data.loc[modelY]
            
            # only necessary b/c bokeh's $index identifier for hovertool
            # pulling integer index despite data being indexed by cellid.
            # if can figure out away to pull cellid strings from index instead,
            # will no longer need this code.
            cells = []
            if self.fair:
                cells = list(set(dataX.index.values.tolist()))
                if cells != list(set(dataY.index.values.tolist())):
                    self.script = 'Problem with form_data_array:'
                    self.div = 'Model x: ' + modelX + 'and Model y: ' + modelY\
                            + ' applied to different cells despite fair check.'
            else:
                cellsX = list(set(dataX.index.values.tolist()))
                cellsY = list(set(dataY.index.values.tolist()))
                if len(cellsX) >= len(cellsY):
                    cells = cellsX
                else:
                    cells = cellsY
            
            
            data = pd.DataFrame({'x_values':dataX[self.measure[0]],\
                                 'y_values':dataY[self.measure[0]],
                                 'cellid':cells})
            print(data)
            
            dat_source = ColumnDataSource(data)
                
            p = figure(x_range=[0,1], y_range=[0,1],x_axis_label=modelX,\
                       y_axis_label=modelY, title=self.measure[0], tools=tools)
                    
            glyph = Circle(x='x_values',y='y_values',size=5,fill_color='navy',\
                           fill_alpha=0.5)
            p.add_glyph(dat_source,glyph)
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
            # TODO: hardcoded self.measure[0] for now, but should incorporate
            #       a for loop somewhere to subplot for each selected measure
            
            # TODO: add significance information (see plot_bar_pretty and randttest
                                                  # in narf_analysis)      
    
            #build new pandas series of stdev values to be added to dataframe
            
            #if want to show more info on tooltip in the future, just need
            #to build an appropriate series to add and then build its tooltip
            #in the create_hover function

            modelnames = self.data.index.levels[0].tolist()
            
            index = range(len(modelnames))
            stdev_col = pd.Series(index=index)
            mean_col = pd.Series(index=index)
            model_col = pd.Series(index=index,dtype=str)
            
            #for each model, find the stdev and mean over the measure values, then
            #assign those values to new Series objects to use for the plot
            i = 0
            for model in modelnames:
                values = self.data[self.measure[0]].loc[model].values
                                       
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
                       'Mean %s'%self.measure[0],title="Mean %s Performance By Model"\
                       %self.measure[0],tools=tools)
            
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
                 measure = 'r_test',extra_cols=['n_parms']):
        PlotGenerator.__init__(self,data,fair,outliers,measure,extra_cols)
            
    def create_hover(self):
        hover_html = """
        <div>
            <span class="hover-tooltip">parameters: $x</span>
        </div>
        <div>
            <span class="hover-tooltip">%s value: $y</span>
        </div>
        """%self.measure[0]
        return HoverTool(tooltips=hover_html)
            
    def generate_plot(self):
        
        tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
                     self.create_hover()]
            
        
        # TODO: Change this to custom chart type? Not quite the same as narf pareto
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
        
        p = BoxPlot(self.data,values=self.measure[0],label='n_parms',\
                        title="Mean Performance (%s) versus Complexity"%self.measure[0],\
                        tools=tools, color='n_parms')
            
        self.script,self.div = components(p)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
"""Defines classes for generating bokeh plots.

PlotGenerator defines the base class from which each plot type extends.
Each plot class takes in a dataframe, a performance measure, and some optional
specifications. The dataframe is converted into a new DataFrame
multi-indexed by cell and model, with a column for each performance measure
as well as any additional columns specified. This way, there is a unique
set of values for each cell and model combination. After the data is converted,
generate_plot can be invoked to store the jsonified script and div
representations of the bokeh plot for the data, which can then be embedded in
an html document.

"""

import math
import itertools

from bokeh.plotting import figure
from bokeh.io import gridplot
from bokeh.embed import components
from bokeh.models import (
        ColumnDataSource, HoverTool, ResizeTool ,SaveTool, WheelZoomTool,
        PanTool, ResetTool, Range1d, FactorRange,
        )
from bokeh.charts import Bar, BoxPlot
from bokeh.models.glyphs import VBar,Circle
import pandas as pd
import numpy as np

# Setting default tools as global variable was causing issues with scatter
# plot. They're included here for copy-paste as needed instead.
#tools = [PanTool(),ResizeTool(),SaveTool(),WheelZoomTool(),ResetTool(),\
#                     self.create_hover()]


class PlotGenerator():
    """Base class for plot generators."""
    
    def __init__(
            self, data, measure, fair=True, outliers=False, extra_cols=[],
            ):
        # Force measure to be interpreted as a list for
        # forward-compatibility with specifying multiple measures.
        if not isinstance(measure, iter):
            self.measure = [str(measure)]
        else:
            self.measure = list(measure)
        self.fair = fair
        self.outliers = outliers
        self.extra_cols = extra_cols
        self.data = self.form_data_array(data)
        
        # Use this inside views function to check whether generate_plot
        # should be invoked.
        if (self.data.size == 0):
            self.emptycheck = True
        else:
            self.emptycheck = False

    def generate_plot(self):
        """Assigns script and div attributes to the plot generator object.
        
        All plot classes should implement some form of this function to ensure
        that script and div are returned to the relevant view function.
        
        Assigns:
        --------
        self.script : JSON data (or a string default/error)
            Bokeh javascript needed by whatever HTML document embeds self.div
        self.div : JSON data (or a string default/error)
            Bokeh javascript used to render a plot from the attached
            datasource.
        
        """
        
        self.script, self.div = ('','')
    
    def create_hover(self):
        """Returns a Bokeh HoverTool() object, possibly with generated 
        toolip HTML.
        
        """
        
        return HoverTool()
    
    def form_data_array(self, data):
        """Formats data into a multi-indexed DataFrame for plotting.
        
        Takes a DataFrame (typically a full NarfResults query) and converts it
        to a new multi-indexed DataFrame (cellid level 0, modelname level 1)
        with a column for each performance measure, plus any other columns
        needed for specific plots.
        If 'outliers' isn't checked, cellid rows containing values that don't
            meet outlier criteria will be removed.
        If 'fair' is checked, cellid rows containing a NaN value in any
            column will be removed.
            
        Returns:
        --------
        newData : Pandas DataFrame w/ multi-index
            Multi-indexed DataFrame containing a series of values for each
            cellid + modelname combination.
        
        See Also:
        ---------
        Narf_Analysis : compute_data_matrix
        
        """
        
        celllist = [
                cell for cell in
                list(set(data['cellid'].values.tolist()))
                ]
        modellist = [
                model for model in
                list(set(data['modelname'].values.tolist()))
                ]
        # Use lists of unique cell and model names to form a multiindex.
        multiIndex = pd.MultiIndex.from_product(
                [celllist,modellist], names=['cellid','modelname'],
                )
        newData = pd.DataFrame(
                index = multiIndex, columns = self.measure+self.extra_cols,
                )
        newData.sort_index()

        for c in celllist:
            for m in modellist:
                dataRow = data.loc[(data.cellid == c) & (data.modelname == m)]
                
                # Add column values for any additional columns specificed
                # in plot class (ex: n_parms for pareto plot)
                if self.extra_cols:
                    for col in self.extra_cols:
                        try:
                            colval = dataRow[col].values.tolist()[0]
                        except Exception as e:
                            # TODO: Is this a good way to do this?
                            #       This requires that all extra columns also 
                            #       have values for every cell/model
                            #       if fair is checked.
                            colval = math.nan
                            print(e)
                        finally:
                            newData[col].loc[c,m] = colval
                
                for meas in self.measure:
                    value = math.nan
                    newData[meas].loc[c,m] = value
                    # If loop hits a continue, value will be left as NaN.
                    # Otherwise, will be assigned a value from data 
                    # after passing all checks.
                    try:
                        value = dataRow[meas].values.tolist()[0]
                    except Exception as e:
                        # Error should mean no value was recorded,
                        # so leave as NaN.
                        # No need to run outlier checks if value is missing.
                        print(e)
                        continue
                    
                    if not self.outliers:
                        # If outliers is false, run a bunch of checks based on
                        # measure and if a check fails, step out of the loop.
                        
                        # Comments for each check are copied from
                        # from Narf_Analysis : compute_data_matrix
                        
                        # "Drop r_test values below threshold"
                        a1 = (meas == 'r_test')
                        b1 = (value < dataRow['r_floor'].values.tolist()[0])
                        a2 = (meas == 'r_ceiling')
                        b2 = (
                            dataRow['r_test'].values.tolist()[0]
                            < dataRow['r_floor'].values.tolist()[0]
                            )
                        a3 = (meas == 'r_floor')
                        b3 = b1
                        if (a1 and b1) or (a2 and b2) or (a3 and b3):
                            continue
                    
                        # "Drop MI values greater than 1"
                        a1 = (meas == 'mi_test')
                        b1 = (value > 1)
                        a2 = (meas == 'mi_fit')
                        b2 = (0 <= value <= 1)
                        if (a1 and b1) or (a2 and not b2):
                            continue
                           
                        # "Drop MSE values greater than 1.1"
                        a1 = (meas == 'mse_test')
                        b1 = (value > 1.1)
                        a2 = (meas == 'mse_fit')
                        b2 = b1
                        if (a1 and b1) or (a2 and b2):
                            continue
                           
                        # "Drop NLOGL outside normalized region"
                        a1 = (meas == 'nlogl_test')
                        b1 = (-1 <= value <= 0)
                        a2 = (meas == 'nlogl_fit')
                        b2 = b1
                        if (a1 and b1) or (a2 and b2):
                            continue
                           
                        # TODO: is this still used? not listed in NarfResults
                        # "Drop gamma values that are too low"
                        a1 = (meas == 'gamma_test')
                        b1 = (value < 0.15)
                        a2 = (meas == 'gamma_fit')
                        b2 = b1
                        if (a1 and b1) or (a2 and b2):
                            continue

                        # TODO: is an outlier check needed for cohere_test
                        #       and/or cohere_fit?
                        
                    # If value existed and passed outlier checks,
                    # re-assign it to the proper DataFrame position
                    # to overwrite the NaN value.
                    newData[meas].loc[c,m] = value

        if self.fair:
            # If fair is checked, drop all rows that contain a NaN value for
            # any column.
            for c in celllist:
                for m in modellist:
                    if newData.loc[c,m].isnull().values.any():
                        newData.drop(c, level='cellid', inplace=True)
                        break
        
        # Swap the 0th and 1st levels so that modelname is the primary index,
        # since most plots group by model.
        newData = newData.swaplevel(i=0, j=1, axis=0)

        # Leaving these in for testing to make sure dropping NaN values
        # is working correctly
        print("was fair checked?")
        print(self.fair)
        print("does the data look different or contain nans?")
        print(newData[self.measure[0]].values)
        
        return newData
    
        
        
class Scatter_Plot(PlotGenerator):
    """Defines the class used to generate a model-comparison scatter plot."""
    
    def __init__(self, data, measure, fair=True, outliers=False):
        PlotGenerator.__init__(self, data, measure, fair, outliers)
        
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
    
        def __init__(self, data, measure, fair=True, outliers=False):
            PlotGenerator.__init__(self, data, measure, fair, outliers)
            
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
                
                # TODO: Looks like some kinds of checks are needed here to
                #       handle NaN values -- getting "Out of range float"
                #       issue with JSON serialization.
                
                
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
            
    def __init__(
            self, data, measure, fair=True, outliers=False,
            extra_cols=['n_parms']
            ):
        PlotGenerator.__init__(self, data, measure, fair, outliers, extra_cols)
            
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
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        
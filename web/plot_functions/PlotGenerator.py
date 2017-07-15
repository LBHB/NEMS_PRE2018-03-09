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

from bokeh.io import gridplot
from bokeh.plotting import figure
#from bokeh.layouts import widgetbox
from bokeh.embed import components
from bokeh.models import (
        ColumnDataSource, HoverTool, ResizeTool ,SaveTool, WheelZoomTool,
        PanTool, ResetTool, Range1d, FactorRange,
        )
from bokeh.charts import BoxPlot
from bokeh.models.glyphs import VBar,Circle
#from bokeh.models.widgets import DataTable, TableColumn
import pandas as pd
import numpy as np

#NOTE: All subclasses of PlotGenerator should be added to the PLOT_TYPES
#      list for use with web interface
PLOT_TYPES = [
        'Scatter_Plot', 'Bar_Plot', 'Pareto_Plot', 'Difference_Plot',
        'Tabular_Plot',
        ]

# Setting default tools as global variable was causing issues with scatter
# plot. They're included here for copy-paste as needed instead.
#tools = [
#    PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
#    ResetTool(), self.create_hover()
#    ]


# Specify the number of columns to use for gridplots
GRID_COLS = 1
# Appearance options for circle glyphs (ex. scatter plot)
CIRCLE_FILL = 'navy'
CIRCLE_SIZE = 5
CIRCLE_ALPHA = 0.5
# Appearance options for virtical bar glyphs (ex. bar plot)
VBAR_FILL = '#FF5740'
VBAR_WIDTH = 0.5
# Position of toolbar around plot area
# (above, below, left, right)
TOOL_LOC = 'above'
# Should the toolbar be inside the axis?
# TODO: check back on bokeh issue. currently cannot combine tool_loc above
#       with tool_stick false due to a conflict with responsive mode.
#       -jacob 7/12/17
TOOL_STICK = True


class PlotGenerator():
    """Base class for plot generators."""
    
    def __init__(
            self, data, measure, fair=True, outliers=False, extra_cols=[],
            ):
        # Force measure to be interpreted as a list for
        # forward-compatibility with specifying multiple measures.
        if isinstance(measure, list):
            self.measure = measure + extra_cols
        else:
            self.measure = [measure] + extra_cols
        self.fair = fair
        self.outliers = outliers
        print("Re-formatting data array...")
        self.data = self.form_data_array(data)
        
        # Use this inside views function to check whether generate_plot
        # should be invoked.
        if (self.data.size == 0):
            self.emptycheck = True
        else:
            self.emptycheck = False
            
        print("Building plot...")
            

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
        
        self.script, self.div = ('Plot type is','not yet implemented')
    
    def create_hover(self):
        """Returns a Bokeh HoverTool() object, possibly with generated 
        toolip HTML.
        
        $ and @ tell the HoverTool() where to pull data values from.
        $ denotes a special column (ex: $index or $x),
            refer to Bokeh docs for a full list.
        @ denotes a field in the data source (ex: @cellid)
        
        """
        
        return HoverTool()
    
    def form_data_array(self, data):
        """Formats data into a multi-indexed DataFrame for plotting.
        
        Takes a DataFrame (built from a full NarfResults query) and converts it
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
                index = multiIndex, columns = self.measure,
                )
        
        newData.sort_index()

        for c in celllist:
            for m in modellist:
                dataRow = data.loc[(data.cellid == c) & (data.modelname == m)]
                
                # Add column values for any additional columns specificed
                # in plot class (ex: n_parms for pareto plot)
                #if self.extra_cols:
                #    for col in self.extra_cols:
                #        try:
                #            colval = dataRow[col].values.tolist()[0]
                #        except Exception as e:
                #            # TODO: Is this a good way to do this?
                #            #       This requires that all extra columns also 
                #            #       have values for every cell/model
                #            #       if fair is checked.
                #            colval = math.nan
                #            print(e)
                #        finally:
                #            newData[col].loc[c,m] = colval
                
                for meas in self.measure:
                    value = np.nan 
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
                        print("No %s recorded for %s,%s"%(meas,c,m))
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

        # Leave these in for testing to make sure dropping NaN values
        # is working correctly
        #print("was fair checked?")
        #print(self.fair)
        #print("does the data look different or contain nans?")
        #print(newData[self.measure[0]].values)

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
        """Iteratively reformats and plots self.data for each cell+model combo.
        
        TODO: Finish this doc
        
        """
        
        plots = []
        modelnames = self.data.index.levels[0].tolist()
        
        # Iterate over a list of tuples representing all unique pairs of models.
        for pair in list(itertools.combinations(modelnames,2)):
            tools = [
                PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), self.create_hover(),
                ]

            modelX = pair[0]
            modelY = pair[1]
            
            dataX = self.data.loc[modelX]
            dataY = self.data.loc[modelY]
            
            # Only necessary b/c bokeh's $index identifier for HoverTool()
            # is pulling an integer index despite data being indexed by cellid.
            # If cellid strings can be pulled from index instead,
            # this code will no longer be needed.
            cells = []
            cellsX = list(set(dataX.index.values.tolist()))
            cellsY = list(set(dataY.index.values.tolist()))
            if self.fair:
                # cellsX and cellsY should be the same if fair was checked
                cells = cellsX
                if cells != cellsY:
                    self.script = 'Problem with form_data_array:'
                    self.div = 'Model x: ' + modelX + 'and Model y: ' + modelY\
                            + ' applied to different cells despite fair check.'
                    return
            else:
                # If fair wasn't checked, use the longer list to avoid errors.
                if len(cellsX) >= len(cellsY):
                    cells = cellsX
                else:
                    cells = cellsY
            
            data = pd.DataFrame({
                    'x_values':dataX[self.measure[0]],
                    'y_values':dataY[self.measure[0]],
                    'cellid':cells,
                    })
            dat_source = ColumnDataSource(data)
                
            p = figure(
                    x_range=[0,1], y_range=[0,1],
                    x_axis_label=modelX, y_axis_label=modelY,
                    title=self.measure[0], tools=tools, responsive=True,
                    toolbar_location=TOOL_LOC, toolbar_sticky=TOOL_STICK,
                    webgl=True,
                    )
            glyph = Circle(
                    x='x_values', y='y_values', size=CIRCLE_SIZE,
                    fill_color=CIRCLE_FILL, fill_alpha=CIRCLE_ALPHA,
                    )
            p.add_glyph(dat_source, glyph)
            p.line([0,1], [0,1], line_width=1, color='black')
            plots.append(p)
    
        # If more than one plot was made (i.e. 2 or more models were selected),
        # put them in a grid.

        if len(plots) == 1:
            singleplot = plots[0]
            self.script,self.div = components(singleplot)
            return
        elif len(plots) > 1:            
            grid = gridplot(
                    plots, ncols=GRID_COLS, responsive=True,
                    )
            self.script,self.div = components(grid)
        else:
            self.script, self.div = (
                    'Error, no plots to display.',
                    'Make sure you selected two models.'
                    )


class Bar_Plot(PlotGenerator):
    """Defines the class used to generate a mean-performance bar plot for
    a model-by-model comparison.
    
    """
    
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
        """Calculates mean and standard deviation for measure(s) by model,
        then generates a bar plot of model vs mean performance.
        
        TODO: Finish this doc.
        
        """

        # Use this for a built-in bar plot instead,
        # but doesn't work with custom hover tool
        #p = Bar(self.data,label='modelname',values=self.measure,agg='mean',\
        #        title='Mean %s Performance By Model'%self.measure,legend=None,\
        #        tools=tools, color='modelname')
        #self.script,self.div = components(p)
        #return
        
        # TODO: hardcoded self.measure[0] for now, but should incorporate
        #       a for loop somewhere to subplot for each selected measure
        
        # TODO: add significance information (see plot_bar_pretty
        #       and randttest in narf_analysis)      

        # build new pandas series of stdev values to be added to dataframe
        # if want to show more info on tooltip in the future, just need
        # to build an appropriate series to add and then build its tooltip
        #in the create_hover function

        modelnames = self.data.index.levels[0].tolist()
        stdev_col = pd.Series(index=modelnames)
        mean_col = pd.Series(index=modelnames)
        #for each model, find the stdev and mean over the measure values, then
        #assign those values to new Series objects to use for the plot
        for model in modelnames:
            values = self.data[self.measure[0]].loc[model]
            stdev = values.std(skipna=True)
            mean = values.mean(skipna=True)
            if (math.isnan(stdev)) or (math.isnan(mean)):
                # If either statistic comes out as NaN, entire column was NaN,
                # so model doesn't have the necessary data.
                continue
            stdev_col.at[model] = stdev
            mean_col.at[model] = mean
            
        newData = pd.DataFrame.from_dict({
                'stdev':stdev_col, 'mean':mean_col,
                })
        # Drop any models with NaN values, since that means they had no
        # performance data for one or more columns.
        newData.dropna(axis=0, how='any', inplace=True)
        if newData.size == 0:
            self.script,self.div = (
                    "Error, no plot to display.",
                    "None of the models contained valid performance data."
                    )
            return
        dat_source = ColumnDataSource(newData)
        
        tools = [
                PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), self.create_hover()
                ]
        xrange = FactorRange(factors=modelnames)
        yrange = Range1d(
                start=0,
                end=(max(newData['mean'])*1.5)
                )
        p = figure(
                x_range=xrange, x_axis_label='Model',
                y_range=yrange, y_axis_label='Mean %s'%self.measure[0],
                title="Mean %s Performance By Model"%self.measure[0],
                tools=tools, responsive=True, toolbar_location=TOOL_LOC,
                toolbar_sticky=TOOL_STICK
                )
        p.xaxis.major_label_orientation=-(np.pi/4)
        glyph = VBar(
                x='index', top='mean', bottom=0, width=VBAR_WIDTH,
                fill_color=VBAR_FILL, line_color='black'
                )
        p.add_glyph(dat_source,glyph)
            
        self.script,self.div = components(p)
            
            
class Pareto_Plot(PlotGenerator):
    """Defines the class used to generate a Bokeh box-plot for mean performance
    versus model complexity.
    
    """
    
    # Always include 'n_parms' as an extra column, since it's required
    # for this plot type.
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
        """TODO: write this doc."""
        
        tools = [
                PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), self.create_hover()
                ]
            
        
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
        
        p = BoxPlot(
                self.data, values=self.measure[0], label='n_parms',
                title="Mean Performance (%s) vs Complexity"%self.measure[0],
                tools=tools, color='n_parms', responsive=True,
                toolbar_location=TOOL_LOC, toolbar_sticky=TOOL_STICK,
                )
            
        self.script,self.div = components(p)
            
            
class Difference_Plot(PlotGenerator):
    def form_data_array(self, data):
        return pd.DataFrame(index=['A','B'], columns=['1','2'])
    # TODO: implement this from NARF
            
    
class Tabular_Plot(PlotGenerator):
    # TODO: implement this from NARF
    def __init__(self, data, measure, fair=True, outliers=False, extra_cols=[]):
        # Use blank measure since tabular has a fixed set of columns
        _measure=[]
        _extra_cols=[
                'r_test', 'mse_test', 'nlogl_test',
                'mi_test', 'cohere_test', 'n_parms',
                ]
        PlotGenerator.__init__(
                self, data=data, measure=_measure, fair=fair,
                outliers=outliers, extra_cols=_extra_cols
                )
        
    def generate_plot(self):
        # After __init__ self.measure should contain everything that
        # was passed in extra_cols
        
        if self.fair:
            self.script, self.div = (
                    "Uncheck 'only fair' ",
                    "to use this plot"
                    )
            return
        
        columns = []
        for m in self.measure:
            if m == 'n_parms':
                columns.append(m)
                break
            mean = 'mean_%s'%m
            median = 'median_%s'%m
            columns.append(mean)
            columns.append(median)
        table = pd.DataFrame(
                    # index = list of model names
                    index=self.data.index.levels[0].tolist(),
                    # columns = list of measures, both mean and median
                    columns=columns
                    )
        self.data.replace(0, np.nan, inplace=True)
        
        for i, model in enumerate(table.index.tolist()):
            for j, meas in enumerate(self.measure):
                series = self.data[meas]
                if j%2 == 0:
                    col = 'mean_%s'%meas
                else:
                    col = 'median_%s'%meas
                    
                if 'n_parms' in meas:
                    table.at[model, 'n_parms'] = (
                            series.loc[model].values.tolist()[0]
                            )
                    break
                else:
                    table.at[model, col] = np.nanmean(series.loc[model])
                    table.at[model, col] = np.nanmedian(series.loc[model])
        
        table.sort_values('mean_r_test', axis=0, ascending=False, inplace=True)
        
        # see pandas style attribute for more options
        positives = [
                'mean_r_test', 'median_r_test',
                'mean_mi_test', 'median_mi_test',
                'mean_cohere_test', 'median_cohere_test',
                ]
        negatives = [
                'mean_mse_test', 'median_mse_test',
                'mean_nlogl_test', 'median_nlogl_test',
                ]
        table.style.highlight_max(subset=positives, axis=0, color='darkorange')
        table.style.highlight_min(subset=negatives, axis=0, color='darkorange')
        self.html = table.to_html(
            index=True, classes="table-hover table-condensed",
            )
        
        #source = ColumnDataSource(table)
        #cols = table.columns.tolist()
        #columns = []
        #for c in cols:
        #    columns.append(
        #            TableColumn(field='index', title='Model')
        #            )
        #    columns.append(
        #            TableColumn(field=c, title=c)
        #            )
        #data_table = DataTable(
        #        source=source, columns=columns, editable=False,
        #        reorderable=True, sortable=True,
        #        )
        
        #self.script, self.div = components(data_table)

            
            
            
            
            
            
            
            
            
            
            
            
            
        

"""Reports which cells have been fitted by which models for a given batch.

Code similar to PlotGenerator class, but different enough that it's
been separated here.

"""

from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import (
        ColumnDataSource, HoverTool, ResizeTool ,SaveTool, WheelZoomTool,
        PanTool, ResetTool, Range1d, FactorRange,
        )
from bokeh.charts import HeatMap
import pandas as pd
import numpy as np


class Status_Report():
    def __init__(self, data):
        self.data = data
        self.batch = data['batch'].iat[0]
    
    def generate_plot(self):
        tools = [
                PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), self.create_hover()
                ]
        p = HeatMap(self.data, x='cellid', y='modelname', values='r_test',
                    stat=None, title=str(self.batch), hover_text='r_test',
                    tools=tools, responsive=True,
                    )
        p.yaxis.major_label_orientation='horizontal'
        p.xaxis.visible = False
        
        self.script, self.div = components(p)
    
    def create_hover(self):

        return HoverTool()
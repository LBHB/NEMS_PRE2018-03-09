"""Reports which cells have been fitted by which models for a given batch.

Code similar to PlotGenerator class, but different enough that it's
been separated here.

"""

from bokeh.embed import components
from bokeh.models import (
        HoverTool, ResizeTool ,SaveTool, WheelZoomTool,
        PanTool, ResetTool,
        )
from bokeh.charts import HeatMap


class Performance_Report():
    def __init__(self, data, batch):
        self.data = data
        self.batch = batch
    
    def generate_plot(self):
        tools = [
                PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), HoverTool()
                ]
        p = HeatMap(self.data, x='cellid', y='modelname', values='r_test',
                    stat=None, title=str(self.batch), hover_text='r_test',
                    tools=tools, responsive=True,
                    )
        p.yaxis.major_label_orientation='horizontal'
        p.xaxis.visible = False
        
        self.script, self.div = components(p)
    
class Fit_Report():
    def __init__(self, data):
        self.data = data
        
    def generate_plot(self):
        tools = [
                PanTool(), ResizeTool(), SaveTool(), WheelZoomTool(),
                ResetTool(), HoverTool(tooltips=[
                        ('modelname','$x'), ('cellid','$y'), ('status','@yn')
                        ])
                ]
        p = HeatMap(
                self.data, x='modelname', y='cellid', values='yn',
                stat=None, tools=tools, responsive=True, hover_text='yn',
                )
        p.yaxis.visible = False
        p.xaxis.visible = False
        
        self.script, self.div = components(p)
        
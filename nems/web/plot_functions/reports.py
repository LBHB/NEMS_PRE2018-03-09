"""Reports which cells have been fitted by which models for a given batch.

Code similar to PlotGenerator class, but different enough that it's
been separated here.

"""

import io

import matplotlib as mpl
import matplotlib.pyplot as plt, mpld3
import seaborn as sns
import numpy as np
import scipy as scp

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
        array = self.data.values
        cols = self.data.columns.tolist()
        rows = self.data.index.tolist()
        # Try upsampling if array dimensions too small to avoid interp
        # note: didn't work very well on tests.
        #if (len(rows) < 20) or (len(cols) < 20):
        #    array = scp.misc.imresize(array, 20.0, interp='nearest')
        xticks = range(len(cols))
        yticks = range(len(rows))
        minor_xticks = np.arange(-0.5, len(cols), 1)
        minor_yticks = np.arange(-0.5, len(rows), 1)
        extent = self.extents(xticks) + self.extents(yticks)
        
        p = plt.figure(figsize=(len(cols),len(rows)/4))
        img = plt.imshow(
                array, aspect='auto', origin='lower', 
                cmap=plt.get_cmap('RdBu'), interpolation='none',
                extent=extent,
                )
        img.set_clim(0, 0.6)
        ax = plt.gca()
        ax.set_ylabel('')
        ax.set_xlabel('Model')
        ax.set_yticks(yticks)
        ax.set_yticklabels(rows, fontsize=8)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=8)
        ax.set_yticks(minor_yticks, minor=True)
        ax.set_xticks(minor_xticks, minor=True)
        ax.grid(b=False)
        ax.grid(which='minor', color='w', linestyle='-', linewidth=0.75)
        cbar = plt.colorbar()
        cbar.set_ticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        cbar.set_ticklabels([
                'Dead', '', '', 'Missing', 'In Progress', 'Not Started',
                'Complete',
                ])
        # Should set the colorbar to sit at the top of the figure (y=1),
        # which it does, but also forces plot way down, which is bad.
        #cax = cbar.ax
        #current_pos = cax.get_position()
        #cax.set_position(
        #        [current_pos.x0, 1, current_pos.width, current_pos.height]
        #        )
        
        
        img = io.BytesIO()
        plt.savefig(img, bbox_inches='tight')
        #html = mpld3.fig_to_html(p)
        plt.close(p)
        img.seek(0)
        self.img_str = img.read()
        
    def extents(self, f):
        # reference:
        # https://bl.ocks.org/fasiha/eff0763ca25777ec849ffead370dc907
        # (calculates the data coordinates of the corners for array chunks)
        delta = f[1] - f[0]
        return [f[0] - delta/2, f[-1] + delta/2]

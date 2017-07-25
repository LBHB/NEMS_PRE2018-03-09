"""
The most common intracellular elecrophysiology program is pClamp, which records
data in a proprietary .abf filetype.  There are a couple different python
modules that import .abf files.  The we will use for this exercise is neo. To
install this package, go to the command line and type:

   pip install neo

We will use the neo.io.AxonIO class to read the file. The example file,
'PVcell3.abf' is a whole-cell current clamp recording of a
parvalbumin-expressing fast-spiking interneuron in an adult mouse brain slice.
Both negative and positive 500ms current pulses were injected to explore the
cell's firing properties.

A common way to show how a neuron responds to input is to plot the firing rate
vs amplitude of the current injected into the cell.  This plot tells us whether
the cell transiently fires, whether the spike rate reaches a maximum at some
high current step, the gain of firing (the slope of the line, in Hz/pA), etc.
To construct a f-i curve we need to first find the number of spikes in each
sweep and the current injection that drove those spikes. Then plot it.
"""
from neo.io import AxonIO
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import pylab as pl
import numpy as np


'''
Preliminary steps to read in the data.
'''
# I commonly use the variable 'fh' as shorthand for 'file handle'. In
# programming parlance, 'handle' is an abstract reference to a resource (in this
# case, the AxonIO instance that allows you to read from the file).
fh = AxonIO(filename='PVcell3.abf')

# The block is a container for the voltage and current data stored in the file.
block = fh.read_block()

# Some information regarding the experiment has been saved in the file. We need
# to parse this information out for the analysis. A block consists of multiple
# sweeps (i.e., ,trials), known as `segments`. Each segment can contain multiple
# signals, known as `analogsignals`. For this particular data file, the number
# of channels, number of timepoints, sampling rate and sampling period are
# identical across all sweeps.
n_channels = len(block.segments[0].analogsignals)
n_sweeps = len(block.segments)
n_timepoints = len(block.segments[0].analogsignals[0].times)
sampling_rate = int(block.segments[0].analogsignals[0].sampling_rate)
sampling_period = 1.0/sampling_rate
sweep_duration = n_timepoints*sampling_period

message = '''The data contains {n_sweeps} sweeps (i.e., trials) acquired from
{n_channels} channels. Each sweep contains {n_timepoints} timepoints sampled at
{sampling_rate} Hz, for a total duration of {sweep_duration} sec.'''

print (message.format(n_sweeps=n_sweeps, n_channels=n_channels,
                     n_timepoints=n_timepoints, sampling_rate=sampling_rate,
                     sweep_duration=sweep_duration))

'''
Exercise 1
----------

Each "sweep" is stored as a "segment" in the datafile. Using the data stored in
the file, create a pandas DataFrame with two columns labeled 'voltage' and
'current'. The DataFrame should be indexed by sweep number and time. The first
five rows of the dataframe should look like the following (where sweep and time
are labels in a hierarchial MultiIndex):

                current    voltage
    sweep time
    0     0.00000      0.0 -48.889160
          0.00001      0.0 -49.011230
          0.00002      0.0 -49.072266
          0.00003      0.0 -48.980713
          0.00004      0.0 -49.041748

The last five rows should look like the following:

                current    voltage
    sweep time
    19    0.99995      0.0 -54.962158
          0.99996      0.0 -54.931641
          0.99997      0.0 -54.748535
          0.99998      0.0 -54.931641
          0.99999      0.0 -55.023193

To read in the data from the first segment, we we can do the following:

    voltage_data = block.segments[0].analogsignals[0]
    current_data = block.segments[0].analogsignals[1]

You can also loop through the segments:

    for segment in block.segments:
        # do something
'''

# Exercise 1 solution here
time = np.arange(0, sweep_duration, sampling_period)
dataframes = []
for segment in block.segments:
    channel1 = segment.analogsignals[0]
    channel2 = segment.analogsignals[1]
    df = pd.DataFrame({'voltage': channel1,
                       'current': channel2}, index=time)
    dataframes.append(df)
	
# Now that we've made our list of dataframe, merge these into a single dataframe
# that we can work with. Here, we pass a new level in the index (the sweep
# number) which will be indexed from 0 ... n_sweeps. We have to specify the
# names for the new hierarchial index. Note that the hierarchial index will
# include the original "time" index for each dataframe, so we have to include
# the name for that as well.
data = pd.concat(dataframes, keys=np.arange(n_sweeps), names=['sweep', 'time'])



'''
Exercise 2
----------
In one figure plot all the voltage traces overlayed in one subplot and all the
current traces that were used to stimulate the cell in the other subplot
'''

# Exercise 2 solution here
pl.figure()
voltage_ax = pl.subplot(121)
current_ax = pl.subplot(122, sharex=voltage_ax)
current_ax.set_xlabel('Time (sec)')
voltage_ax.set_xlabel('Time (sec)')
current_ax.set_ylabel('Current (pA)')
voltage_ax.set_ylabel('Voltage (uV)')

# data stores time and trials in "long" format
voltage=data.loc[:,'voltage'] # gives series of voltages
y=voltage.unstack(level='sweep')

# The simplest way to plot the data is to unstack it so that we have a 2D array
# of sweep x time
unstacked_data = data.unstack(level='time')

# loc[row, (column)]  # use parens to disambiguate what gets grouped as row vs. column
#unstacked_data.loc[:,('voltage',0)].head()

time = unstacked_data['voltage'].columns
voltage_ax.plot(time, unstacked_data['voltage'].T, '-')
current_ax.plot(time, unstacked_data['current'].T, '-')

pl.tight_layout()
pl.show(False)

'''
Exercise 3
----------
In an new figure plot sweeps 0, 5, 10 and 15.
'''

# Exercise 3 solution here

'''
Exercise 4
----------
Take sweep 0 and find the time of the spike at the end of the current trace.
Plot the trace with a red dot at the time and amplitude of the spike.
'''

# Exercise 4 solution here
sweep_data=data.loc[0,'voltage']
time=sweep_data.argmax()
voltage=sweep_data.max()

# find all threshold crossings
threshold = -20
sweep_data[sweep_data>=threshold].head()


'''
Exercise 5
----------
Ok, that was easy because there was only one spike. But what if there is more
than one spike in the data? For example, take a look at sweep 11. Yikes! Write
a function that takes the voltage data for sweep 11 and counts the number of
spikes.
'''

# Exercise 5 solution here

'''
Exercise 6
----------
Use this function to count the number of spikes for each sweep.
'''

# Exercise 6 solution here

'''
Exercise 7
----------
The value of the current at time == 0.1 for each sweep gives you the current
amplitude. Extract that value for each sweep.
'''

# Exercise 7 solution here

'''
Exercise 8
----------
Now that you have the number of spikes per sweep and the current amplitude on
each sweep, plot the current vs number of spikes for the cell.
'''

# Exercise 8 solution here

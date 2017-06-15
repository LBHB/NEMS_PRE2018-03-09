# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:54:47 2016

@author: Tim

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


Here is what I'd like to do:
1. Download the neo module from for your OS
2. import neo and use it to import the data from 'PVcell3.abf
3. In one figure plot all the voltage traces overlayed in the upper subplot and
   all the current traces that were used to stimulate the cell in the lower
   subplot.
4. In another figure plot each sweep (voltage) that had a positive current pulse
   in a different subplot
5. Count the spikes in each trace and plot a red circle over their peaks in the
   previous figure note: to make this work I used scipy.signal.argrelmax, but
   there may be a better option. I filtered the signal first, found the relative
   maxima, then retained only the maxima above a threshold where action
   potential peaks would be.
6. Make an f-i curve for this cell.  An f-i curve is a cells respone to positive
   current pulses.  It typically has firing frequency (in Hz) on the y axis and
   the size of the current step on the x-axis.
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

print message.format(n_sweeps=n_sweeps, n_channels=n_channels,
                     n_timepoints=n_timepoints, sampling_rate=sampling_rate,
                     sweep_duration=sweep_duration)

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
# We are going to read in each segment (i.e., sweep) individually and create a
# list of dataframes. Each dataframe will be indexed by time. Since the
# timepoints are the same for all dataframes, just create the 1D array for the
# timepoints once and then reuse it each time we create a new dataframe.
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
pl.figure()
voltage_ax = pl.subplot(121)
current_ax = pl.subplot(122, sharex=voltage_ax)
current_ax.set_xlabel('Time (sec)')
voltage_ax.set_xlabel('Time (sec)')
current_ax.set_ylabel('Current (pA)')
voltage_ax.set_ylabel('Voltage (uV)')

# The simplest way to plot the data is to unstack it so that we have a 2D array
# of sweep x time
unstacked_data = data.unstack(level='time')
time = unstacked_data['voltage'].columns
voltage_ax.plot(time, unstacked_data['voltage'].T, '-')
current_ax.plot(time, unstacked_data['current'].T, '-')

pl.tight_layout()

'''
Exercise 3
----------
In an new figure plot sweeps 0, 5, 10 and 15.
'''
pl.figure(figsize=(12, 12))
n_rows = 2
n_cols = 2

# The loc command is very powerful. It lets us pull out  multiple sections based
# on the hierarchial index. Here, we are telling it to pull out the data where
# the first level of the index matches 0, 5, 10 and 15.
data_subset = data.loc[[0, 5, 10, 15]]

# Now, we group the data by the sweep level in the index. Groupby will return
# the value of the current group and the subset of the data for that group.
for i, (sweep, sweep_df) in enumerate(data_subset.groupby(level='sweep')):
    # This is a check to see if we're creating the very first axes. If we are,
    # then just create it. If an axes already exists from the previous iteration
    # of the for loop, we want to make sure that we share the x and y
    # dimensions across all axes. Note that when pl.subplot(..., sharex=ax) is
    # evaluated, ax is still pointing to the old instance of ax from the prior
    # loop. Once the function is evaluated, it returns a new instance of ax that
    # we use for plotting the current sweep.
    if i == 0:
        ax = pl.subplot(n_rows, n_cols, i+1)
    else:
        ax = pl.subplot(n_rows, n_cols, i+1, sharex=ax, sharey=ax)

    time = sweep_df.index.get_level_values('time')
    ax.plot(time, sweep_df['voltage'], '-')
    ax.set_title('Sweep {}'.format(sweep))
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('Voltage (uV)')

pl.tight_layout()

'''
Exercise 4
----------
Take sweep 0 and find the time of the spike at the end of the current trace.
Plot the trace with a red dot at the time and amplitude of the spike.
'''

# Pull out the first sweep and look only at the voltage column. Although the
# `sweep` level of the multiindex disappears, sweep_data will still be indexed
# by time. The method argmax() returns the index where the maximum value occurs.
sweep_data = data.loc[0, 'voltage']
time = sweep_data.argmax()
voltage = sweep_data.max()

pl.figure()
pl.plot(sweep_data.index.get_level_values('time'), sweep_data, 'k-')
pl.plot(time, voltage, 'ro')
pl.xlabel('Time (sec)')
pl.ylabel('Voltage (uV)')

'''
Exercise 5
----------
Ok, that was easy because there was only one spike. But what if there is more
than one spike in the data? For example, take a look at sweep 11. Yikes! Write
a function that takes the voltage data for sweep 11 and counts the number of
spikes.
'''

# This is really a very simple algorithm. The voltage traces are very clean, so
# we can just define a threshold. Anytime the voltage trace exceeds that
# threshold, we know a spike occurred.  There are multiple ways to detect how
# often the voltage exceeds threshold. However, this is one of the simplest
# methods I can think of.
def count_spikes(voltage, threshold=-20):
    # Create a boolean mask indicating which voltage points exceed the
    # threshold.  The mask will look like [False, False, True, True, False,
    # False ...]. This means at index 2 and 3, there was a spike.
    mask = voltage >= threshold

    # We can count the number of spikes by counting how many times we see a
    # False -> True transition. We can use the np.diff function for this.
    # However:
    #   np.diff([False, False, True, True, False, False, ...])
    # gives us:
    #   [False, True, False, True, False, ...]
    # Essentially what is happening here is that it is preserving the data-type
    # of the array. That means that when two consecutive values have no change
    # (i.e., False -> False or True -> True), the result is False. However, when
    # there is a change (i.e., False -> True or True -> False), the result is
    # True. However, we want to know only when we go from False -> True. So, we
    # need to convert the array to an integer datatype. This gives np.diff more
    # options. So, now the mask will look like (where True has been converted to
    # 1 and False to 0):
    #   [0, 0, 1, 1, 0, 0, ...]
    # Calling np.diff then gives us:
    #   [0, 1, 0, -1, 0]
    # Note that we now have 1 for all 0 (i.e., False) to 1 (True) transitions
    # (i.e., the beginning of the spike). The end of the spike is marked with a
    mask = mask.astype('i')

    # Now, all we have to do is count the number of elements where the
    # transition is 1. That gives us the number of times the voltage crossed
    # from below to above the thresholds cutoff.
    transitions = np.diff(mask)
    return np.sum(transitions == 1)

n_spikes_sweep_11 = count_spikes(data.loc[11, 'voltage'])

'''
Exercise 6
----------
Use this function to count the number of spikes for each sweep.
'''

# Groupby will provide the subset of the dataframe to the function and create a
# new dataframe from the result provided by the function.
n_spikes = data.groupby(level='sweep')['voltage'].apply(count_spikes)

'''
Exercise 7
----------
The value of the current at time == 0.1 for each sweep gives you the current
amplitude. Extract that value for each sweep.
'''
current = data.xs(0.1, level='time')['current']

'''
Exercise 8
----------
Now that you have the number of spikes per sweep and the current amplitude on
each sweep, plot the current vs number of spikes for the cell.
'''
pl.figure()
pl.plot(current, n_spikes, 'ko-')
pl.xlabel('Current (pA)')
pl.ylabel('Number of spikes')

# Brad's code for saving the figures
for number in pl.get_fignums():
    figure = pl.figure(number)
    filename = 'week_7_solution_{}.png'.format(number)
    figure.savefig(filename)

pl.show()

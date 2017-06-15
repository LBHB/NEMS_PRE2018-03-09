# -*- coding: utf-8 -*-

# The data consist of 2 csv files from a study of the effects of loud noise
# exposure on the auditory brainstem response (ABR)  in Veterans. ABR data.csv
# provides the results from the ABR measures. This datafile consists of the
# subject id, the stimulus level in dB p-pe SPL , the frequency of the stimulus
# in kHz, the amplitude (in uV) of the wave 1 peak, and the amplitude of the
# wave 1 trough. Each stimulus is repeated 2 times, resulting in 2 rows of data
# for each stimulus frequency/level combination. Subject data.csv provides
# additional information about each subject including age, gender, study group,
# and whether or not they report tinnitus. You will need to use groupby for this
# problem.

import pandas as pd
import pylab as pl

abr_filename = r'ABR data.csv'
subjectdata_filename = r'Subject data.csv'
output_filename = r'merged_data.csv'

# Read in the csv files
abr_data = pd.io.parsers.read_csv(abr_filename)
abr_data = abr_data[['subject_id', 'level', 'frequency', 'wave_1_peak', 'wave_1_trough']]

subject_data = pd.io.parsers.read_csv(subjectdata_filename)
subject_data = subject_data[['SubjectID', 'Age', 'Gender', 'StudyGroup','Tinnitus']]

# Problem 1: Take the average of the 2 replications for each ABR stimulus
# frequency/level combination.

# Each combination of subject_id, level and frequency has two replications. We
# want to group by these three columns and compute the average of the remaining
# two columns (wave_1_peak and wave_1_trough). See below for why we use
# `as_index=False`.
grouping = ['subject_id', 'level', 'frequency']
abr_data_average = abr_data.groupby(grouping, as_index=False).mean()

# Problem 2: Merge the two data sets into a new dataframe based on subject id
# and save it as a new csv file. Use to_csv() for this. Note that the subject id
# column has a slightly different name in each original csv file.

# There are two approaches to combining these dataframes. To understand these
# approaches, you need to understand what it means for a dataframe to have an
# index. The index is essentially a column or set of columns that specify unique
# labels for each row. It's easy to move entries between the index and the
# columns using the DataFrame.set_index and DataFrame.reset_index methods.
# DataFrame.groupby returns a dataframe indexed by the group labels by default.
# This behavior can be modified. Compare the following:
abr_data_average_v1 = abr_data.groupby(grouping).mean()
abr_data_average_v2 = abr_data.groupby(grouping, as_index=False).mean()

# Based on the way I set up this merge, we do not want subject_id to be in the
# index for abr_data_average. Hence, the reason I used `as_index=False` in
# problem 1.
merged_data_v1 = pd.merge(abr_data_average, subject_data,
                          left_on=['subject_id'], right_on=['SubjectID'],
                          how='right')

# This is a second approach to merging the dataframes. If the "right" dataframe
# has a column or set of columns that form a unique index (i.e., the
# combinations of the columns form a unique value for each row), then we can use
# this approach. First, we specify the index for the subject_data dataframe and
# tell it to modify the dataframe in-place. If we set inplace=False, then the
# original dataframe would not be modified. Instead, a copy of the dataframe
# would be made, the operation performed on the dataframe, and the copy of the
# dataframe returned.
subject_data.set_index(['SubjectID'], inplace=True, verify_integrity=True)
merged_data_v2 = abr_data_average.join(subject_data, on=['subject_id'],
                                       how='right')

# Note that in approach 1, we have a column for 'subject_id' and 'SubjectID'. In
# approach 2, only 'subject_id' is retained. Note that we are missing some
# entries for subject IDs, hence the reason we use a right join. We only want to
# analyze ABR data for which we have information about the subject.
merged_data = merged_data_v2

# Problem 3: Create a new column in your merged dataframe that contains the
# absolute amplitude of # ABR wave 1 (wave 1 peak – wave 1 trough).
merged_data['wave_1_amplitude'] = merged_data['wave_1_peak'] \
    -merged_data['wave_1_trough']

# If your dataframe is extremely large (e.g., millions of rows) it may be faster
# to use a special operation that leverages the numexpr library (it streamlines
# the process of transfering data back and forth between the CPU and memory).
merged_data['wave_1_amplitude'] = merged_data.eval('wave_1_peak-wave_1_trough')

# Problem 4: Calculate the mean and standard deviation for wave 1 amplitude (for
# a 4 kHz 110 dB p-pe SPL stimulus) for those who report tinnitus vs. those who
# don’t and for males vs. females. You'll find males and people with tinnitus
# have slightly lower mean amplitudes.

# There are several ways to pull out only the 4 kHz, 110 dB p-pe SPL data.
# First, we could simply compute the mean and standard deviation for all
# frequencies and levels and then pull out the data we want. There are at least
# three ways to pull out the subset of rows we want.
grouping = ['Gender', 'Tinnitus', 'frequency', 'level']
wave_1_metrics = merged_data.groupby(grouping) \
    ['wave_1_amplitude'].agg(['mean', 'std'])

# Now, demonstrate several ways of pulling out the subset of rows we want
metrics_subset_v1 = wave_1_metrics \
    .xs(4, level='frequency') \
    .xs(110, level='level')

idx = pd.IndexSlice
metrics_subset_v2 = wave_1_metrics.loc[idx[:, :, 4, 110], :]

metrics_subset_v3 = wave_1_metrics.query('(frequency == 4) & (level == 110)')

# Or, we can just keep only the rows we want before computing the metrics.
mask = (merged_data.frequency == 4) & (merged_data.level == 110)
subset = merged_data.loc[mask]
metrics_subset_v4 = subset.groupby(['Gender', 'Tinnitus']) \
    ['wave_1_amplitude'].agg(['mean', 'std'])

# Problem 5: Plot wave 1 amplitude vs. age for a 4 kHz 110 dB p-pe SPL stimulus.
# Use dots rather than a line graph. There doesn’t seem to be a strong
# relationship between age and wave 1 amplitude in this dataset.
pl.plot(subset['Age'], subset['wave_1_amplitude'], 'ko')
pl.xlabel('Subject age (years)')

# Matplotlib allows us to embed special notation known as MathTEX into our
# script so we can render greek symobls and equations between dollar signs using
# the notation.
pl.ylabel(r'Wave 1 amplitude ($\mu V$)')

# Problem 6: Plot mean wave 1 amplitude for each study group for a 4 kHz
# stimulus at 80, 90, 100, and 110 dB p-pe SPL on the same graph (level on the
# x-axis and wave 1 amplitude on the y-axis). Also on the same graph, plot the
# wave 1 amplitudes at these levels for each individual subject. Each group
# should be represented with a different color. Note that 2 of the groups have
# lower wave 1 amplitudes at 110 than the other 2 groups.

# Note that this operation returns a series (not a dataframe) indexed by
# StudyGrpoup and level.
subset = merged_data.query('frequency == 4')
mean_wave_1_amplitude = subset.groupby(['StudyGroup', 'level']) \
    ['wave_1_amplitude'].mean()

# There are several ways we can work with this series. One aproach is to group
# by the StudyGroup portion of the index (if the data is in the index, we pass
# this information via the level parameter to groupby).
#pl.figure()
#for group, group_data in mean_wave_1_amplitude.groupby(level='StudyGroup'):
#    # group_data is a subset of mean_wave_1_amplitude and is still a series. We
#    # can extract the values in the index for a specified level.
#    x = group_data.index.get_level_values('level')
#    pl.plot(x, group_data, '-', lw=2, label=group)

# An alternate (and possibly more familiar approach) is to just convert to a
# dataframe. Calling reset_index on a series with a hierarchial index converts
# each level of the index to a column in the dataframe. Important! For a series,
# the values for the series will be saved to a column whose name is the original
# name of the series (e.g., check the `mean_wave_1_amplitude.name` attribute).
# If no name is set (this will depend on the series of operations that led to
# the creation of the series), then the column will have a name of '0'.
mean_wave_1_amplitude_df = mean_wave_1_amplitude.reset_index()

# Note that we no longer specify level since the StudyGroup information is a
# column rather than a level in the index. Here, I am also using the default
# order of colors Matplotlib cycles through for each plot. I save the color used
# for each study group to a dictionary named group_colors so I can be sure to
# plot the individuals with the appropriate colors.
pl.figure()
group_colors = {}
for group, group_data in mean_wave_1_amplitude_df.groupby('StudyGroup'):
    # The plot command returns a list of lines created by the plot. Since only
    # one line is created, the length of the list is 1. We can unpack the first
    # entry in this 1-element list into a variable called 'l' by appending a
    # comma. This is equivalent to doing:
    #   lines = pl.plot(...)
    #   l = lines[0]

    # Parameters to plot
    # -----------------------
    # label
    #   specifies the text that will appear in the legend for that line.
    # zorder
    #   Specifies how it is plotted with relation to other items on the plot.
    #   Items with a higher zorder will be plotted on top of items with a lower
    #   zorder. This ensures that the averages are plotted *on top* of the
    #   individual traces.
    # lw
    #   Thickness of the line.
    l, = pl.plot(group_data['level'], group_data['wave_1_amplitude'], '-',
                 label=group, lw=4, zorder=1)
    group_colors[group] = l.get_color()

# Plot the individual lines
for group, group_data in subset.groupby('StudyGroup'):
    for _, subject_data in group_data.groupby('subject_id'):
        # Setting the label to '_nolegend_' is required for my particular
        # combination of Matplotlib and Pandas to prevent a legend entry from
        # showing up for each individual line. I'm not sure why this is the case
        # or what changed (your versions of Matplotlib/Pandas may not have this
        # "feature").
        # We want the individual trace to be plotted with the color used for the
        # average trace representing the StudyGroup the individual is in.
        pl.plot(subject_data['level'], subject_data['wave_1_amplitude'], '-',
                color=group_colors[group], lw=0.5, zorder=0, label='_nolegend_')

pl.xlabel('Stimulus level (dB p-pe SPL)')
pl.ylabel('ABR wave 1 amplitude (peak to peak, $\mu V$)')
pl.legend(loc='upper left')
pl.show()

# Problem 7: Plot the mean wave 1 amplitude for each study group for a 1, 3, 4,
# and 6 kHz stimulus at 110 dB p-pe SPL. Note that 2 of the study groups have
# lower wave 1 amplitudes than the other 2 groups at each of the stimulus
# frequencies.

# Given what we have learned above, this is pretty straightforward. We just
# swap frequency and level in various operations.
subset = merged_data.query('level == 110')
mean_wave_1_amplitude = subset.groupby(['StudyGroup', 'frequency']) \
    ['wave_1_amplitude'].mean()
mean_wave_1_amplitude_df = mean_wave_1_amplitude.reset_index()

pl.figure()
for group, group_data in mean_wave_1_amplitude_df.groupby('StudyGroup'):
    pl.plot(group_data['frequency'], group_data['wave_1_amplitude'], '-',
            color=group_colors[group], lw=4, label=group)

pl.xlabel('Stimulus frequency (kHz)')
pl.ylabel('ABR wave 1 amplitude (peak to peak, $\mu V$)')
pl.legend(loc='lower right')
pl.show()

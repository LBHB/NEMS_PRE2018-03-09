# -*- coding: utf-8 -*-

#The data consist of 2 csv files from a study of the effects of loud noise 
#exposure on the auditory brainstem response (ABR)  in Veterans. ABR data.csv 
#provides the results from the ABR measures. This datafile consists of the subject
#id, the stimulus level in dB p-pe SPL , the frequency of the stimulus in kHz, the
#amplitude (in uV) of the wave 1 peak, and the amplitude of the wave 1 trough. Each
#stimulus is repeated 2 times, resulting in 2 rows of data for each stimulus 
#frequency/level combination. Subject data.csv provides additional information about
#each subject including age, gender, study group, and whether or not they report 
#tinnitus. You will need to use groupby for this problem.

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



# Problem 1: Take the average of the 2 replications for each ABR stimulus frequency/level combination.

## PROBLEM 1 ANSWER
abr_avgd=abr_data.groupby(['subject_id','level','frequency'], as_index=False).mean()

# OR
#abr_avgd=abr_data.groupby(['subject_id','level','frequency']).mean()
#abr_avgd.reset_index(inplace=True)


# Problem 2: Merge the two data sets into a new dataframe based on subject id and 
# save it as a new csv file. Use to_csv() for this. Note that the subject id column has a slightly 
# different name in each original csv file.

## PROBLEM 2 ANSWER
full_data=pd.merge(subject_data,abr_avgd,left_on='SubjectID',right_on='subject_id',how='inner')



# Problem 3: Create a new column in your merged dataframe that contains the 
# absolute amplitude of # ABR wave 1 (wave 1 peak – wave 1 trough).

## PROBLEM 3 ANSWER
full_data['wave_1_diff']=full_data['wave_1_peak']-full_data['wave_1_trough']


# Problem 4: Calculate the mean and standard deviation for wave 1 amplitude (for 
# a 4 kHz 110 dB p-pe SPL stimulus) for those who report tinnitus vs. those who 
# don’t and for males vs. females. You'll find males and people with tinnitus have 
# slightly lower mean amplitudes.

## PROBLEM 4 ANSWER
gender_data=full_data.query('frequency==4 & level==110')
gender_mean=gender_data.groupby(['Gender','Tinnitus'])['wave_1_diff','wave_1_peak'].agg(['mean','std'])

# Problem 5: Plot wave 1 amplitude vs. age for a 4 kHz 110 dB p-pe SPL stimulus.
# Use dots rather than a line graph. There doesn’t seem to be a strong 
# relationship between age and wave 1 amplitude in this dataset.

## PROBLEM 5 ANSWER
pl.plot(gender_data['Age'], gender_data['wave_1_diff'], 'k.')
pl.xlabel('Age (years)')

# Problem 6: Plot mean wave 1 amplitude for each study group for a 4 kHz stimulus at 80, 
# 90, 100, # and 110 dB p-pe SPL on the same graph (level on the x-axis and wave 1 amplitude 
# on the # y-axis). Also on the same graph, plot the wave 1 amplitudes at these levels for each 
# individual subject. Each group should be represented with a different color. Note that 2 of 
# the groups have lower wave 1 amplitudes at 110 than the other 2 groups.


## PROBLEM 6 ANSWER
level_data=full_data.query('frequency==4')
level_mean=level_data.groupby(['StudyGroup','level'])['wave_1_diff'].mean()
level_mean=level_mean.reset_index()
pl.figure()
group_colors = {}
for group, group_data in level_mean.groupby('StudyGroup'):
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
    l, = pl.plot(group_data['level'], group_data['wave_1_diff'], '-o',
                 label=group, lw=2, zorder=1)
    group_colors[group] = l.get_color()
    
for group, group_data in level_data.groupby('StudyGroup'):
    for _, subject_data in group_data.groupby('subject_id'):
        # Setting the label to '_nolegend_' is required for my particular
        # combination of Matplotlib and Pandas to prevent a legend entry from
        # showing up for each individual line. I'm not sure why this is the case
        # or what changed (your versions of Matplotlib/Pandas may not have this
        # "feature").
        # We want the individual trace to be plotted with the color used for the
        # average trace representing the StudyGroup the individual is in.
        pl.plot(subject_data['level'], subject_data['wave_1_diff'], '-',
                color=group_colors[group], lw=0.5, zorder=0, label='_nolegend_')


pl.legend(group_colors.keys(),loc='upper left')
pl.xlabel('Level (dB SQL)')
pl.ylabel('ABR wave 1 peak-trough')

# Problem 7: Plot the mean wave 1 amplitude for each study group for a 1, 3, 4,
# and 6 kHz stimulus at 110 dB p-pe SPL. Note that 2 of the study groups have
# lower wave 1 amplitudes than the other 2 groups at each of the stimulus 
# frequencies.

## PROBLEM 7 ANSWER
freq_data=full_data.query('level==110')
freq_mean=freq_data.groupby(['StudyGroup','frequency'])['wave_1_diff'].mean()
freq_mean=freq_mean.reset_index()
pl.figure()
group_colors = {}
for group, group_data in freq_mean.groupby('StudyGroup'):
    l, = pl.plot(group_data['frequency'], group_data['wave_1_diff'], '-o',
                 label=group, lw=2, zorder=1)
    group_colors[group] = l.get_color()
    
#pl.legend(group_colors.keys(),loc='lower right')
pl.legend(loc='lower right')
pl.xlabel('Frequency (kHz)')
pl.ylabel('ABR wave 1 peak-trough')
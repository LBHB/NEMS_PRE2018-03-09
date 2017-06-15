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

abr_filename = r'C:\Users\vhaporbramhn\Desktop\ABR data.csv'
subjectdata_filename = r'C:\Users\vhaporbramhn\Desktop\Subject data.csv'
output_filename = r'C:\Users\vhaporbramhn\Desktop\merged_data.csv'

# Read in the csv files

abr_data = pd.io.parsers.read_csv(abr_filename)
abr_data = abr_data[['subject_id', 'level', 'frequency', 'wave_1_peak', 'wave_1_trough']]

subject_data = pd.io.parsers.read_csv(subjectdata_filename)
subject_data = subject_data[['SubjectID', 'Age', 'Gender', 'StudyGroup','Tinnitus']]

# Problem 1: Take the average of the 2 replications for each ABR stimulus frequency/level combination.

## PROBLEM 1 ANSWER

# Problem 2: Merge the two data sets into a new dataframe based on subject id and 
# save it as a new csv file. Use to_csv() for this. Note that the subject id column has a slightly 
# different name in each original csv file.

## PROBLEM 2 ANSWER

# Problem 3: Create a new column in your merged dataframe that contains the 
# absolute amplitude of # ABR wave 1 (wave 1 peak – wave 1 trough).

## PROBLEM 3 ANSWER

# Problem 4: Calculate the mean and standard deviation for wave 1 amplitude (for 
# a 4 kHz 110 dB p-pe SPL stimulus) for those who report tinnitus vs. those who 
# don’t and for males vs. females. You'll find males and people with tinnitus have 
# slightly lower mean amplitudes.

## PROBLEM 4 ANSWER

# Problem 5: Plot wave 1 amplitude vs. age for a 4 kHz 110 dB p-pe SPL stimulus.
# Use dots rather than a line graph. There doesn’t seem to be a strong 
# relationship between age and wave 1 amplitude in this dataset.

## PROBLEM 5 ANSWER

# Problem 6: Plot mean wave 1 amplitude for each study group for a 4 kHz stimulus at 80, 
# 90, 100, # and 110 dB p-pe SPL on the same graph (level on the x-axis and wave 1 amplitude 
# on the # y-axis). Also on the same graph, plot the wave 1 amplitudes at these levels for each 
# individual subject. Each group should be represented with a different color. Note that 2 of 
# the groups have lower wave 1 amplitudes at 110 than the other 2 groups.

## PROBLEM 6 ANSWER

# Problem 7: Plot the mean wave 1 amplitude for each study group for a 1, 3, 4,
# and 6 kHz stimulus at 110 dB p-pe SPL. Note that 2 of the study groups have
# lower wave 1 amplitudes than the other 2 groups at each of the stimulus 
# frequencies.

## PROBLEM 7 ANSWER

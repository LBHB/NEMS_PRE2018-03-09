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

abr_data = abr_data.groupby(['subject_id', 'frequency', 'level'])[['wave_1_peak','wave_1_trough']].mean().reset_index()

# Problem 2: Merge the two data sets into a new dataframe based on subject id and 
# save it as a new csv file. Use to_csv() for this. Note that the subject id column has a slightly 
# different name in each original csv file.

subject_data.set_index('SubjectID', inplace=True)

abr_data['SubjectID'] = abr_data['subject_id']
abr_data.set_index('subject_id', inplace=True)

merged_data = abr_data.join(subject_data, on='SubjectID')
merged_data.to_csv(output_filename, index=False)

# Problem 3: Create a new column in your merged dataframe that contains the 
# absolute amplitude of # ABR wave 1 (wave 1 peak – wave 1 trough).

merged_data['wave_1_amplitude']=merged_data['wave_1_peak']-merged_data['wave_1_trough']

# Problem 4: Calculate the mean and standard deviation for wave 1 amplitude (for 
# a 4 kHz 110 dB p-pe SPL stimulus) for those who report tinnitus vs. those who 
# don’t and for males vs. females. You'll find males and people with tinnitus have 
# slightly lower mean amplitudes.

mask_110_4K = (merged_data['level'] == 110) & (merged_data['frequency'] == 4)
merged_data_110_4K = merged_data[mask_110_4K]

tinnitus_mean = merged_data_110_4K.groupby('Tinnitus')[['wave_1_amplitude']].mean()
tinnitus_std = merged_data_110_4K.groupby('Tinnitus')[['wave_1_amplitude']].std()

gender_mean = merged_data_110_4K.groupby('Gender')[['wave_1_amplitude']].mean()
gender_std = merged_data_110_4K.groupby('Gender')[['wave_1_amplitude']].std()

print 'Mean' + str(tinnitus_mean)
print 'std' + str(tinnitus_std)

print 'Mean' + str(gender_mean)
print 'std' + str(gender_std)

# Problem 5: Plot wave 1 amplitude vs. age for a 4 kHz 110 dB p-pe SPL stimulus.
# Use dots rather than a line graph. There doesn’t seem to be a strong 
# relationship between age and wave 1 amplitude in this dataset.

pl.figure(1)
pl.plot(merged_data_110_4K['wave_1_amplitude'],merged_data_110_4K['Age'], 'ko')
pl.xlabel(u'Wave 1 Amplitude (uV)')
pl.ylabel('Age (years)')
pl.show()

# Problem 6: Plot mean wave 1 amplitude for each study group for a 4 kHz stimulus at 80, 
# 90, 100, # and 110 dB p-pe SPL on the same graph (level on the x-axis and wave 1 amplitude 
# on the # y-axis). Also on the same graph, plot the wave 1 amplitudes at these levels for each 
# individual subject. Each group should be represented with a different color. Note that 2 of 
# the groups have lower wave 1 amplitudes at 110 than the other 2 groups.

mask_4K = (merged_data['frequency']==4)
merged_data_4K = merged_data[mask_4K]

group_colors = {
    'Veteran High Noise': 'black',
    'Veteran Low Noise': 'gray',
    'Non-Veteran No Noise': 'red',
    'Non-Veteran Firearms': 'blue',
}

pl.figure(2)
for group_name, group_data in merged_data_4K.groupby('StudyGroup'):
    color = group_colors[group_name]
    group_mean = group_data.groupby('level')[['wave_1_amplitude']].mean()
    group_mean.reset_index(inplace=True)
    
    ax1=pl.subplot(111)
    ax1.plot(group_mean['level'], group_mean['wave_1_amplitude'], color=color,
            lw=6, zorder=1, label=group_name)
    ax1.set_xlabel('Level (dB p-pe SPL)')
    ax1.set_ylabel(u'Wave 1 amplitude (uV)')
    ax1.legend(loc='upper left')
    
    for subject_id, subject_data in group_data.groupby('SubjectID'):
        ax1.plot(subject_data['level'], subject_data['wave_1_amplitude'],
                color=color, lw=1, alpha = 0.6, zorder=0)
pl.show()

# Problem 7: Plot the mean wave 1 amplitude for each study group for a 1, 3, 4,
# and 6 kHz stimulus at 110 dB p-pe SPL. Note that 2 of the study groups have
# lower wave 1 amplitudes than the other 2 groups at each of the stimulus 
# frequencies.

mask_110 = (merged_data['level']==110)
merged_data_110 = merged_data[mask_110]

pl.figure(3)
for group_name, group_data in merged_data_110.groupby('StudyGroup'):
    color = group_colors[group_name]
    group_mean = group_data.groupby('frequency')[['wave_1_amplitude']].mean()
    group_mean.reset_index(inplace=True)
    
    ax2=pl.subplot(111)
    ax2.plot(group_mean['frequency'], group_mean['wave_1_amplitude'], color=color,
            lw=6, zorder=1, label=group_name)
    ax2.set_xlabel('Frequency (kHz)')
    ax2.set_ylabel(u'Wave 1 amplitude (uV)')
    ax2.legend(loc='upper left')
pl.show()


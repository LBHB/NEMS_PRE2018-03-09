# Python class -- WEEK 5
#
# Processing spiking data recorded during presentation of temporally
# orthogonal ripple combinations (TORCs). TORCs are stimulus designed
# for efficient white noise analysis of the auditory system. The basic
# idea is to play a lot of complex random sounds while recording the
# activity of an auditory neuron. You then find the average sound that
# evokes an increase in spiking activity
#
# Goals: 
# 1. Visualize TORC stimulus spectrograms 
# 2. Plot spike raster, showing the time of spike events aligned in
#    time to the spectrogram
# 3. Plot peristimulus time histogram (PSTH) response to the TORCs,
#    i.e., the time-varying firing rate averaged across presentations
#    of the TORC stimulus.

import numpy as np
import pylab as pl
import matplotlib as mp

import scipy.io
import scipy.signal

# The data: 

# Spike data were recorded from a single neuron in primary auditory
# cortex during 2 repetitions of 30 different TORC stimuli, each 2
# seconds long and with 0.5 sec of silence before and after the
# sound. These TORCs consist of the same spectro-temporal pattern
# repeated 4 times a second. So each 2-sec stimulus effectively
# contains cycles of the same sound. The first cycle drives onset
# transients, so usually it is discarded, leaving 7 cycles of
# "steady-state" stimuluation on each trial.

# load contents of Matlab data file
#data = scipy.io.loadmat('tor_data_btn026d-a1.mat',chars_as_strings=True)
data = scipy.io.loadmat('tor_data_por073b-b1.mat',chars_as_strings=True)

# parse into relevant variables

# spectrogram of TORC stimuli. 15 frequency bins X 300 time samples X 30 different TORCs
stim=data['stim']
FrequencyBins=data['FrequencyBins'][0,:]
stimFs=data['stimFs'][0,0]
StimCyclesPerSec=data['StimCyclesPerSec'][0,0]

# response matrix. sampled at 1kHz. value of 1 means a spike occured
# in a particular time bin. 0 means no spike. shape: 3000 time bins X 2
# repetitions X 30 different TORCs

resp=data['resp']
respFs=data['respFs'][0,0]

# each trial is (PreStimSilence + Duration + PostStimSilence) sec long
Duration=data['Duration'][0,0] # Duration of TORC sounds
PreStimSilence=data['PreStimSilence'][0,0]
PostStimSilence=data['PostStimSilence'][0,0]


# 1. Because the fastest stimulus modulations were 50Hz, the stimulus
# can be stored at low resolution (100 Hz). However, to align with the
# response (stored at 1000 Hz), we need to resample the stimulus.  Use
# the scipy.signal.resample() command to resample the temporal axis of
# the stimulus spectrogram from 100 Hz to 1000 Hz

# 2. display the stimulus spectrogram from an entire single trial,
# label time and frequency axes apprpriately

# 3. snip out the first and second 250-ms segements of the stimulus to
# convince yourself that the stimulus is infact repeating.  save a the
# first 250-ms snip for alignment with the response

# 4. trim the first 0.75 sec and last 0.5 sec from the response matrix
# to remove the silent periods and then reshape so that the remaining
# 7 cycles per trial are treated as repetitions. a great opportunity
# to use the reshape command with the -1 option!

# 5. plot spectrogram aligned with the raster for a few different
# TORCs (I recommend TORC 0, 7, & 9). 
# inelegant approach: display the raster using imshow. 
# elegant approach: find spike times and use the plot command to 
# display points for each spike event

# 6. average the rasters across trials and downsample by a factor of 5 to
# plot the PSTH for each TORC response


import numpy as np
import pylab as pl
import matplotlib as mp

import scipy.io

def get_fft(fs, waveforms):
    '''
    Calculate the frequency response for the provided waveforms

    Parameters
    ----------
    fs : float
        Sampling frequency (in Hz)
    waveforms : n-dimensional array
        Set of waveforms where the last axis (i.e., dimension) is time.

    Returns
    -------
    frequencies : 1D array
        Array of frequencies
    psd : n-dimensional array
        Normalized power spectral density (i.e., the frequency response in units
        of V per Hz). All but the final dimension map to the original dimensions
        of the `waveforms` array. The final dimension is frequency.  For
        example, if `waveforms` is a 3D array with the dimensions corresponding
        to repetition, microphone, time, then `psd` will be a 3D array with
        dimensions corresponding to repetition, microphone, frequency.
    phase : Phase of the response

    Example
    -------
    To plot the frequency response of the microphone (remember that waveforms is
    a 3D array with repetition, microphone and time as the dimensions):
    >>> waveforms = np.load('microphone_data.npy')
    >>> fs = 200000
    >>> frequencies, psd = get_fft(fs, waveforms)
    >>> mean_psd = np.mean(psd, axis=0)
    >>> pl.loglog(frequencies, psd[0], 'k-')    # cheap microphone
    >>> pl.loglog(frequencies, psd[1], 'r-')    # expensive microphone
    >>> pl.xlabel('Frequency (Hz)')
    >>> pl.ylabel('Power (Volts)')
    '''
    n_time = waveforms.shape[-1]
    frequencies = np.fft.rfftfreq(n_time, fs**-1)
    csd = np.fft.rfft(waveforms)/n_time
    psd = 2*np.abs(csd)/np.sqrt(2.0)
    phase = np.angle(csd)
    return frequencies, psd, phase
    

# Problem 5: In order to isolate the EFR component coming from hair cells, we
# applied ouabain to the round window. Ouabain kills auditory nerve fibers without 
# disrupting hair cell function, eliminating the response from the auditory
# nerve and beyond, and leaving only the response from the hair cells. Compute the 
# response from the 'After Ouabain' condition as you did for the 'Control' condition
# in Problem 3, and plot it on the same graph using a different color.
# Observe that the response is now flat across modulation frequency, consistent
# with a response from a single source (presumably the hair cells)

## PROBLEM 5 ANSWER
sig_amp=np.zeros((mod_count,2),dtype=np.float)
sig_phase=np.zeros((mod_count,2),dtype=np.float)
noise=np.zeros((mod_count,2),dtype=np.float)

for ii in range(0,mod_count):
    for jj in range(0,2):
        [sig_amp[ii,jj],sig_phase[ii,jj],noise[ii,jj]]=efr_resp(fs,electrode_nm[:,ii,jj],modulation_frequencies[ii])

pl.figure()
pl.plot(modulation_frequencies,sig_amp,'-')
pl.plot(modulation_frequencies,noise,'--')
pl.show(False)


# PART 6 and beyond is a bit tricky to understand, so it's extra optional :-)

# Problem 6: The latency of an unknown system can be ustimated using something
# called group delay. This can be estimated as the slope of the phase vs. 
# modulation frequency function. To compute the group delay, first plot the phase:
#   1) Get an array of the phase at the modulation frequency for each stimulus.
#   2) Plot the phase vs modulation frequency for both the control and ouabain 
#       conditions. Notice how the phase jumps for the control condition
#       This happens because phase is ambiguous (if you have a 1 kHz sine wave,
#       its period is 1 millisecond, so you can't tell the difference between a 
#       delay of 0 ms and 1 ms, 2 ms, etc).


## PROBLEM 6 ANSWER
pl.figure()
pl.plot(modulation_frequencies,sig_phase,'-')
pl.legend(conditions)
pl.ylabel('phase (rad)')
pl.xlabel('modulation frequency')
pl.show(False)


# Problem 7: If you have sampled using fine enough modulation frequency steps,
# you can remove the amibguity by "unwrapping" the phase. We will use np.unwrap
# to do this. This function looks for indexes where the difference in phase between
# adjacent points is greater than a half a period (pi radians), and adds or subtracts 
# one full period (2*pi) to make the difference less than a half period. 
# Continuing the steps to estimate group delay:
#   3) call np.unwrap on the phase plotted in part 6 to unwrap the phase
#   4) plot the unwrapped phase

## PROBLEM 7 ANSWER
pl.figure()
pl.plot(modulation_frequencies,np.unwrap(sig_phase,axis=0),'-')
pl.legend(conditions)
pl.ylabel('unwrapped phase (rad)')
pl.xlabel('modulation frequency')
pl.show(False)


# Problem 8: Compute the slope of the unwrapped phase to find the group delay:
#   5) Compute the slope of the curve
#       slope = diff(unwrapped_phase)/(difference_between_modulation_frequency_steps)        
#       difference_between_modulation_frequency_steps is 60 Hz in this case
#   6) Turn the slope into group delay in milleseconds. The slope is in units
#       of radians * seconds, convert it to group delay by:
#       group_delay = [slope / (2*pi)] * 1000 * -1
#   Plot the group delay for both conditions. Notice how the group delay goes from 
#   a curve with big peaks (due to the response from multiple neural centers 
#   combining with different delays) to a flat line (because there is only one source 
#   left, the hair cells).

## PROBLEM 8 ANSWER
d_freq=np.diff(modulation_frequencies,axis=0)
d_phase=np.diff(np.unwrap(sig_phase,axis=0),axis=0)
c_freq=d_freq/2+modulation_frequencies[0:-1]

slope=d_phase/d_freq[...,np.newaxis]

pl.figure()
pl.plot(c_freq,slope,'-')
pl.legend(conditions)
pl.ylabel('group delay (rad)')
pl.xlabel('modulation frequency')
pl.show(False)

group_delay = (slope / (2*np.pi)) * 1000 * -1
pl.figure()
pl.plot(c_freq,group_delay,'-')
pl.legend(conditions)
pl.ylabel('group delay (ms)')
pl.xlabel('modulation frequency')
pl.show(False)

import os
import json
import pandas as pd
import numpy as np


class Signal():

    def __init__(self, **kwargs):
        ''' 
        A Signal is a convenient class for slicing, averaging, jackknifing,
        truncating, splitting, saving, and loading tabular data that is stored 
        in a CSV file + a JSON metadata sidecar. This class is intended to be 
        useful for loading a dataset, dividing it into estimation and 
        validation subsets, selecting parts of data where a condition is true,
        concatenating Signals together, and other common data wrangling tasks. 

        DATA ACCESS:
        Generally speaking, you won't access a Signal's internal ._matrix data
        directly (it is immutable anyway), but instead call a function that 
        will present the data to you in an expected format. For example:

        sig = nems.signal.load('/path/to/signal')
        ... = sig.as_single_trial()               # Get all of the channels 
        ... = sig.as_single_trial(chans=[0,3,7])  # Get channels 0, 3 and 7
        ... = sig.as_single_trial(reps=[5,6])     # Get trials 5 and 6 only

        # Also useful:
           .as_repetition_matrix()
           .as_average_trial()

        DATA SUBSETS:
        You have two choices when creating a data subset:
        # 1) Get a new signal with the non-matching data excised
          .excised(condition_function)

        # 2) NaN out all data that does not meet a particular condition, use:
          .where(condition_function)

        MODIFIED SIGNAL CREATION:
        It's very common to want to create a new signal from an existing signal.
        You may do that with the following functions:

        .normalized()         # Create a normalized version of the signal
        .split_by_reps(fraction)
        .split_by_time(fraction)
        .jackknifed_by_reps(nsplits, split_idx)
        .jackknifed_by_time(nsplits, split_idx)


        FILE FORMAT: 
        A CSV file should have one row per instant in time, and each column
        should be a different "channel" of the signal. Channels can represent
        whatever dimension you want, such as an auditory frequency, X/Y/Z 
        coordinates for motion, or voltage and current levels for a neuron. 
        It is common for Signals to have multiple channels, because it is 
        common for a tuple of data to be measured at the same instant.
       
        The JSON file specifies optional attributes for the Signal, such as:
           .name       The name of the signal
           .recording  The name of the recording session of this signal
           .chans      A list of the names of each of the channels
           .fs         Frequency of sampling [Hz]
           .nreps      The number of equal-length repetitions to divide
                       the time series into, if applicable. 
           .meta       A catch-all data structure for anything else you want
        
        You may also augment this JSON with other information that describes
        the experimental conditions under which that the data was observed.
        
        Example: to instantiate a Signal object...
        TODO      
        '''
        self._matrix = kwargs['matrix']
        self._matrix.flags.writeable = False  # Make it immutable
        self.name = kwargs['name']
        self.recording = kwargs['recording']
        self.chans = kwargs['chans']
        self.fs = kwargs['fs']
        self.nreps = int(kwargs['nreps']) if 'nreps' in kwargs else 1
        self.meta = kwargs['meta'] if 'meta' in kwargs else None

        (T, C) = self._matrix.shape
        if T < C:
            raise ValueError(('Matrix dims incorrect: (T, C) = '
                              + str((T, C))
                              + '; failing because we expected a long time'
                              + ' series but found T < C'))
        self.nchans = C
        self.ntimes = T

        # Cached properties for speed; their use is however optional
        self.max = np.nanmax(self._matrix, axis=0)
        self.min = np.nanmin(self._matrix, axis=0)
        self.mean = np.nanmean(self._matrix, axis=0)
        self.var = np.nanvar(self._matrix, axis=0)
        self.std = np.nanstd(self._matrix, axis=0)

        if type(self.name) is not str:
            raise ValueError('Name of signal must be a string:'
                             + str(self.name))

        if type(self.recording) is not str:
            raise ValueError('Name of recording must be a string:'
                             + str(self.recording))

        if self.chans and type(self.chans) is not list:
            types_are_str = [(True if c is str else False) for c in self.chans]
            if not all(types_are_str):
                raise ValueError('Chans must be a list of strings:'
                                 + str(self.chans))

        if self.fs < 0:
            raise ValueError('fs of signal must be a positive number:'
                             + str(self.fs))

        if self.nreps < 1:
            raise ValueError('nreps must be a positive integer.'
                             + str(self.nreps))

        self.ntimes_per_rep = T / self.nreps # Not actually an int yet

        if int(self.ntimes_per_rep) != self.ntimes_per_rep :
            raise ValueError('ntimes / nreps must be an integer!'
                             + str(self.nreps))

        self.ntimes_per_rep = int(self.ntimes_per_rep) # Now an int

        if type(self._matrix) is not np.ndarray:
            raise ValueError('matrix must be a np.ndarray:'
                             + type(self._matrix))

    def save(self, dirpath, fmt=None):
        ''' 
        Save this signal to a CSV + JSON sidecar. If desired, you may 
        use optional parameter fmt (for example, fmt='%1.3e') 
        to alter the precision of the floating point matrices.
        '''
        filebase = self.recording + '_' + self.name
        basepath = os.path.join(dirpath, filebase)
        csvfilepath = basepath + '.csv'
        jsonfilepath = basepath + '.json'
        # df = pd.DataFrame(self.as_single_trial())  # 10x slower than savetxt:
        if fmt:
            np.savetxt(csvfilepath,
                       self.as_single_trial(),
                       delimiter=", ",
                       fmt=fmt)
        else:
            np.savetxt(csvfilepath,
                       self.as_single_trial(),
                       delimiter=", ")
        obj = {'name': self.name,
               'chans': self.chans,
               'fs': self.fs,
               'nreps': self.nreps,
               'meta': self.meta}
        with open(jsonfilepath, 'w') as f:
            json.dump(obj, f)
        return (csvfilepath, jsonfilepath)

    @staticmethod
    def load(basepath):
        '''
        Loads the CSV & JSON files at absepath returns a Signal() object.

        Example: If you want to load
           /tmp/sigs/gus027b13_p_PPS_resp-a1.csv
           /tmp/sigs/gus027b13_p_PPS_resp-a1.json
        then give this function
           /tmp/sigs/gus027b13_p_PPS_resp-a1
        '''
        csvfilepath = basepath + '.csv'
        jsonfilepath = basepath + '.json'
        # Weirdly, numpy is 10x slower than read_csv (pandas):
        # mat = np.loadtxt(csvfilepath, delimiter=", ")
        mat = pd.read_csv(csvfilepath, header=None).values
        mat = mat.astype('float')
        with open(jsonfilepath, 'r') as f:
            js = json.load(f)
            s = Signal(name=js['name'],
                       chans=js['chans'],
                       recording=js['recording'],
                       fs=js['fs'],
                       nreps=js['nreps'],
                       meta=js['meta'],
                       matrix=mat)
            return s

    @staticmethod
    def list_signals(directory):
        '''
        Returns a list of all CSV/JSON signal files found in DIRECTORY, 
        Paths are relative, not absolute. 
        '''
        files = os.listdir(directory)
        just_fileroot = lambda f: os.path.splitext(os.path.basename(f))[0]
        csvs = [just_fileroot(f) for f in files if f.endswith('.csv')]
        jsons = [just_fileroot(f) for f in files if f.endswith('.json')]
        overlap = set.intersection(set(csvs), set(jsons))
        return list(overlap)

    def as_single_trial(self):
        '''
        Returns a numpy matrix of Time x Chans
        TODO: kwargs: chans=None, reps
        '''
        return self._matrix.copy()

    def as_repetition_matrix(self):
        '''
        Returns a numpy matrix of Time x Reps x Chans
        TODO: kwargs: chans=None, reps=None
        '''
        m = self._matrix.copy()
        return m.reshape(self.ntimes_per_rep, self.nreps, self.nchans)

    def as_average_trial(self):
        '''
        Returns a matrix of Time x Chans after averaging all of the 
        repetitions together.
        TODO: kwargs: chans=None
        '''
        m = self.as_repetition_matrix()
        return np.nanmean(m, axis=1)

    def _modified_copy(self, m):
        '''
        For internal use when making various immutable copies of this signal.
        '''
        new_obj = Signal(name=self.name,
                         chans=self.chans,
                         recording=self.recording,
                         fs=self.fs,
                         nreps=self.nreps,
                         meta=self.meta,
                         matrix=m)
        return new_obj

    def normalized_by_mean(self):
        """ Returns a new signal, same as this one, but shifted to
        have zero mean and unity variance on each channel."""
        m = self._matrix
        m_normed = (m - m.mean(0)) / m.std(0)
        return self._modified_copy(m_normed)

    def normalized_by_bounds(self):
        """ Returns a new signal, same as this one, but shifted so 
        that the signal will range between 0 and 1 on each channel."""
        m = self._matrix
        m_normed = (m - m.min(0)) / m.ptp(0)    
        return self._modified_copy(m_normed)

    def split_by_reps(self, fraction):
        '''
        Returns a tuple of two signals split at fraction (rounded to the nearest
        repetition) of the original signal. If you had 10 reps of T time samples
        samples, and split it at fraction=0.81, this would return (A, B) where
        A is the first eight reps and B are the last two reps.
        '''        
        split_rep = min(self.nreps - 1, max(1, round(self.nreps * fraction)))
        m = self.as_repetition_matrix()
        left = m[:, 0:split_rep, :]
        (lt, lr, lc) = left.shape
        right = m[:, split_rep:, :]
        (rt, rr, rc) = right.shape
        l = Signal(name=self.name,
                   recording=self.recording,
                   chans=self.chans,
                   nreps=split_rep,
                   fs=self.fs,
                   meta=self.meta,
                   matrix=left.reshape(-1, self.nchans))
        r = Signal(name=self.name,
                   recording=self.recording,
                   chans=self.chans,
                   nreps=(self.nreps - split_rep),
                   fs=self.fs,
                   meta=self.meta,
                   matrix=right.reshape(-1, self.nchans))
        return (l, r)

    def split_by_time(self, fraction):
        '''
        Returns a tuple of 2 new signals; because this may split
        one of the repetition unevenly, it sets the nreps to 1 in both of
        the new signals.
        '''
        m = self.as_single_trial()
        split_idx = max(1, int(self.ntimes * fraction))
        left = m[0:split_idx, :]
        right = m[split_idx:, :]        
        l = Signal(name=self.name,
                   recording=self.recording,
                   chans=self.chans,
                   nreps=1,
                   fs=self.fs,
                   meta=self.meta,
                   matrix=left)
        r = Signal(name=self.name,
                   recording=self.recording,
                   chans=self.chans,
                   nreps=1,
                   fs=self.fs,
                   meta=self.meta,
                   matrix=right)
        return (l, r)

    def jackknifed_by_reps(self, nsplits, split_idx):
        '''
        Returns a new signal, with entire reps NaN'd out. If nreps is not
        an integer multiple of nsplits, an error is thrown. 
        '''
        ratio = (self.nreps / nsplits)
        if ratio != int(ratio):
            raise ValueError('nreps must be an integer multiple of nsplits:'
                             + str(ratio))
        m = self._matrix.copy()
        m[:, :, split_idx] = float('NaN')
        return self._modified_copy(m)

    def jackknifed_by_time(self, nsplits, split_idx):
        '''
        Returns a new signal, with some data NaN'd out based on its position
        in the time stream. split_idx is indexed from 0; if you have 20 splits,
        the first is #0 and the last is #19. 
        '''
        n_time_elements = self.ntimes * self.nreps
        splitsize = int(n_time_elements / nsplits)
        if splitsize < 1:
            raise ValueError('Too many jackknifes? Splitsize was: '
                             + str(splitsize))
        split_start = split_idx * splitsize
        if split_idx == nsplits - 1:
            split_end = n_time_elements
        else:
            split_end = (split_idx + 1) * splitsize
        m = self.as_single_trial().copy()
        m[split_start:split_end, :] = float('NaN')
        return self._modified_copy(self.single_to_multi_trial(m))

    def append_timeseries(self, other):
        """ TODO """
        if not type(other) == type(self):
            raise ValueError('append_timeseries needs another Signal object.')
        pass

    def append_reps(self, other):
        """ TODO """
        if not type(other) == type(self):
            raise ValueError('append_reps needs another Signal object.')
        pass

    def append_channels(self, other):
        """ TODO """
        if not type(other) == type(self):
            raise ValueError('append_reps needs another Signal object.')

    def where(self, condition):
        """Returns a new signal, with data that does not meet the condition
        NaN'd out."""
        # TODO
        pass



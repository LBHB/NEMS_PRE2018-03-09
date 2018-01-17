import os
import json
import pandas as pd
import numpy as np


class Signal():

    def __init__(self, **kwargs):
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

    def split_at_rep(self, fraction):
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

    def split_at_time(self, fraction):
        '''
        Returns a tuple of 2 new signals; because this may split
        one of the repetitions unevenly, it sets the nreps to 1 in both of
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

    def jackknifed_by_reps(self, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with entire reps NaN'd out. If nreps is not
        an integer multiple of nsplits, an error is thrown. 
        Optional argument 'invert' causes everything BUT the jackknife to be NaN.
        '''
        ratio = (self.nreps / nsplits)
        if ratio != int(ratio) or ratio < 1:
            raise ValueError('nreps must be an integer multiple of nsplits:'
                             + str(ratio))
        ratio = int(ratio)
        m = self.as_repetition_matrix()       
        if not invert:
            m[:, split_idx:split_idx+ratio, :] = float('NaN')
        else:
            mask = np.ones_like(m, np.bool)
            mask[:, split_idx:split_idx+ratio, :] = 0
            m[mask] = float('NaN')
        return self._modified_copy(m.reshape(-1, self.nchans))

    def jackknifed_by_time(self, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with some data NaN'd out based on its position
        in the time stream. split_idx is indexed from 0; if you have 20 splits,
        the first is #0 and the last is #19.
        Optional argument 'invert' causes everything BUT the jackknife to be NaN.
        '''
        splitsize = int(self.ntimes / nsplits)
        if splitsize < 1:
            raise ValueError('Too many jackknifes? Splitsize was: '
                             + str(splitsize))
        split_start = split_idx * splitsize
        if split_idx == nsplits - 1:
            split_end = self.ntimes
        else:
            split_end = (split_idx + 1) * splitsize        
        m = self.as_single_trial().copy()    
        if not invert:
            m[split_start:split_end, :] = float('NaN')
        else:
            mask = np.ones_like(m, np.bool)
            mask[split_start:split_end, :] = 0
            m[mask] = float('NaN')
        return self._modified_copy(m.reshape(-1, self.nchans))

    def append_signal(self, other_signal):
        '''
        Returns a new signal that is a copy of this one with other_signal
        appended to the end. Requires that other_signal have the 
        same number of channels, repetition length.
        '''
        if not type(other_signal) == type(self):
            raise ValueError('append_signal needs another Signal object.')
        if not other_signal.fs == self.fs:
            raise ValueError('Cannot append signal with different fs.')
        if not len(other_signal.chans) == len(self.chans):
            raise ValueError('Cannot append signal with different nchans.')
        m = np.concatenate((self._matrix, other_signal._matrix), axis=0)    
        return Signal(name=self.name,
                      recording=self.recording,
                      chans=self.chans,
                      nreps=self.nreps + other_signal.nreps,
                      fs=self.fs,
                      meta=self.meta,
                      matrix=m)

    def combine_channels(self, other_signal):
        ''' 
        Combines other_signal into this one as a new set of channels.
        Requires that both signals be from the same recording and have the 
        same number of time samples. 
        '''
        if not type(other_signal) == type(self):
            raise ValueError('combine_channels needs another Signal object.')
        if not other_signal.fs == self.fs:
            raise ValueError('Cannot combine signals with different fs.')
        if not other_signal.ntimes == self.ntimes:
            raise ValueError('Cannot combine signals with different ntimes')
        m = np.concatenate((self._matrix, other_signal._matrix), axis=1)    
        return Signal(name=self.name,
                      recording=self.recording,
                      chans=self.chans + other_signal.chans,
                      fs=self.fs,
                      meta=self.meta,
                      matrix=m)


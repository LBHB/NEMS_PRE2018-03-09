import os
import json
import pandas as pd
import numpy as np


class Signal():

    def __init__(self, fs, matrix, name, recording, chans, nreps=1, meta=None):
        self._matrix = matrix
        self._matrix.flags.writeable = False  # Make it immutable
        self.name = name
        self.recording = recording
        self.chans = chans
        self.fs = fs
        self.nreps = nreps
        self.meta = meta

        # Verify that we have a long time series
        (C, T) = self._matrix.shape
        if T < C:
            m = 'Incorrect matrix dimensions: (C, T) is {}. ' \
                'We expect a long time series, but T < C'
            raise ValueError(m.format((C, T)))

        self.nchans = C
        self.ntimes = T

        # Cached properties for speed; their use is however optional
        self.channel_max = np.nanmax(self._matrix, axis=-1, keepdims=True)
        self.channel_min = np.nanmin(self._matrix, axis=-1, keepdims=True)
        self.channel_mean = np.nanmean(self._matrix, axis=-1, keepdims=True)
        self.channel_var = np.nanvar(self._matrix, axis=-1, keepdims=True)
        self.channel_std = np.nanstd(self._matrix, axis=-1, keepdims=True)

        if not isinstance(self.name, str):
            m = 'Name of signal must be a string: {}'.format(self.name)
            raise ValueError(m)

        if not isinstance(self.recording, str):
            m = 'Name of recording must be a string: {}'.format(self.recording)
            raise ValueError(m)

        if self.chans and type(self.chans) is not list:
            types_are_str = [(True if c is str else False) for c in self.chans]
            if not all(types_are_str):
                raise ValueError('Chans must be a list of strings:'
                                 + str(self.chans))

        if self.fs < 0:
            m = 'Sampling rate of signal must be a positive number. Got {}.'
            raise ValueError(m.format(self.fs))

        if self.nreps < 1:
            m = 'Number of repetitions must be a positive integer. Got {}.'
            raise ValueError(m.format(self.nreps))

        self.ntimes_per_rep = T / self.nreps # Not actually an int yet

        if int(self.ntimes_per_rep) != self.ntimes_per_rep :
            raise ValueError('ntimes / nreps must be an integer!'
                             + str(self.nreps))

        self.ntimes_per_rep = int(self.ntimes_per_rep) # Now an int

        if type(self._matrix) is not np.ndarray:
            raise ValueError('matrix must be a np.ndarray:'
                             + type(self._matrix))

    def save(self, dirpath, fmt='%.18e'):
        '''
        Save this signal to a CSV + JSON sidecar. If desired, you may
        use optional parameter fmt (for example, fmt='%1.3e')
        to alter the precision of the floating point matrices.
        '''
        filebase = self.recording + '_' + self.name
        basepath = os.path.join(dirpath, filebase)
        csvfilepath = basepath + '.csv'
        jsonfilepath = basepath + '.json'

        np.savetxt(csvfilepath, self.as_continuous(), delimiter=",", fmt=fmt)

        with open(jsonfilepath, 'w') as fh:
            json.dump(self._get_attributes(), fh)
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
            print(js)
            s = Signal(name=js['name'],
                       chans=js.get('chans', None),
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

    def as_continuous(self):
        '''
        Return data as a 2D array of channel x time
        '''
        return self._matrix.copy()

    def as_trials(self):
        '''
        Return data as a 3D array of trial x channel x time
        '''
        new_shape = self.nreps, self.nchans, -1
        return self._matrix.reshape(new_shape)

    def as_average_trial(self):
        '''
        Return data as a 2D array of channel x time averaged across trials
        '''
        m = self.as_trials()
        return np.nanmean(m, axis=0)

    def _get_attributes(self):
        md_attributes = ['name', 'chans', 'fs', 'nreps', 'meta', 'recording']
        return {name: getattr(self, name) for name in md_attributes}

    def _modified_copy(self, data, **kwargs):
        '''
        For internal use when making various immutable copies of this signal.
        '''
        attributes = self._get_attributes()
        attributes.update(kwargs)
        return Signal(matrix=data, **attributes)

    def normalized_by_mean(self):
        '''
        Returns a copy of this signal with each channel normalized to have a
        mean of 0 and standard deviation of 1.
        '''
        m = self._matrix
        m_normed = (m - self.channel_mean) / self.channel_std
        return self._modified_copy(m_normed)

    def normalized_by_bounds(self):
        '''
        Returns a copy of this signal with each channel normalized to the range
        [-1, 1]
        '''
        m = self._matrix
        ptp = self.channel_max - self.channel_mean
        m_normed = (m - self.channel_min) / ptp - 1
        return self._modified_copy(m_normed)

    def split_at_rep(self, fraction):
        '''
        Returns a tuple of two signals split at fraction (rounded to the nearest
        repetition) of the original signal. If you had 10 reps of T time samples
        samples, and split it at fraction=0.81, this would return (A, B) where
        A is the first eight reps and B are the last two reps.
        '''
        # Ensure that the split occurs in the repetition range [1, -1]
        split_rep = round(self.nreps*fraction)
        split_rep = np.clip(split_rep, 1, self.nreps-1)

        data = self.as_trials()
        ldata = data[:split_rep].reshape(self.nchans, -1)
        rdata = data[split_rep:].reshape(self.nchans, -1)

        lsignal = self._modified_copy(data=ldata, nreps=split_rep)
        rsignal = self._modified_copy(data=rdata,
                                           nreps=self.nreps-split_rep)
        return lsignal, rsignal

    def split_at_time(self, fraction):
        '''
        Returns a tuple of 2 new signals; because this may split one of the
        repetitions unevenly, it sets the nreps to 1 in both of the new signals.
        '''
        split_idx = max(1, int(self.ntimes * fraction))

        data = self.as_continuous()
        ldata = data[..., :split_idx]
        rdata = data[..., split_idx:]

        lsignal = self._modified_copy(ldata, nreps=1)
        rsignal = self._modified_copy(rdata, nreps=1)

        return lsignal, rsignal

    def jackknifed_by_reps(self, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with entire reps NaN'd out. If nreps is not an
        integer multiple of nsplits, an error is thrown.  Optional argument
        'invert' causes everything BUT the jackknife to be NaN.
        '''
        ratio = (self.nreps / nsplits)
        if ratio != int(ratio) or ratio < 1:
            m = 'nreps must be an integer multiple of nsplits, got {}'
            raise ValueError(m.format(ratio))

        ratio = int(ratio)
        m = self.as_trials().copy()
        if not invert:
            m[split_idx:split_idx+ratio] = np.nan
        else:
            mask = np.ones_like(m, dtype=np.bool)
            mask[split_idx:split_idx+ratio] = 0
            m[mask] = np.nan
        return self._modified_copy(m.reshape(self.nchans, -1))

    def jackknifed_by_time(self, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with some data NaN'd out based on its position
        in the time stream. split_idx is indexed from 0; if you have 20 splits,
        the first is #0 and the last is #19.
        Optional argument 'invert' causes everything BUT the jackknife to be NaN.
        '''
        splitsize = int(self.ntimes / nsplits)
        if splitsize < 1:
            m = 'Too many jackknifes? Splitsize was {}'
            raise ValueError(m.format(splitsize))

        split_start = split_idx * splitsize
        if split_idx == nsplits - 1:
            split_end = self.ntimes
        else:
            split_end = (split_idx + 1) * splitsize

        m = self.as_continuous().copy()
        if not invert:
            m[..., split_start:split_end] = np.nan
        else:
            mask = np.ones_like(m, dtype=np.bool)
            mask[:, split_start:split_end] = 0
            m[mask] = np.nan
        return self._modified_copy(m.reshape(self.nchans, -1))

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
        m = np.concatenate((self._matrix, other_signal._matrix), axis=-1)
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
        m = np.concatenate((self._matrix, other_signal._matrix), axis=0)
        return Signal(name=self.name,
                      recording=self.recording,
                      chans=self.chans + other_signal.chans,
                      fs=self.fs,
                      meta=self.meta,
                      matrix=m)


import os
import json
import pandas as pd
import numpy as np


class Signal:

    def __init__(self, fs, matrix, name, recording, chans=None, epochs=None,
                 meta=None):
        '''
        Parameters
        ----------
        ... TODO
        epochs : {None, DataFrame}
            DataFrame with two columns ('start_index', 'end_index') denoting the
            start and end time of the trials in the recording. Can contain extra
            columns (e.g., the file it came from, an id for each trial, etc.)
            that may eventually be used later.

        ... TODO
        '''
        self._matrix = matrix
        self._matrix.flags.writeable = False  # Make it immutable
        self.name = name
        self.recording = recording
        self.chans = chans
        self.fs = fs
        self.epochs = epochs
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
            attributes = self._get_attributes()
            # Be sure to convert the dataframe to a dictionary that can be
            # handled by JSON
            attributes['epochs'] = attributes['epochs'].to_json()
            json.dump(attributes, fh)
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
        mat = pd.read_csv(csvfilepath, header=None).values
        mat = mat.astype('float')
        with open(jsonfilepath, 'r') as f:
            js = json.load(f)
            s = Signal(name=js['name'],
                       chans=js.get('chans', None),
                       recording=js['recording'],
                       fs=js['fs'],
                       epochs=pd.read_json(js['epochs']),
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
        Return data as a 3D array of channel x trial x time

        If trials are of uneven length, pads shorter trials with NaN. All trials
        are aligned to the start.
        '''
        if self.epochs is None:
            raise ValueError('Cannot reshape into trials')

        samples = self.epochs['end_index'] - self.epochs['start_index']

        n_samples = int(samples.max())
        n_trials = len(self.epochs)
        n_channels = self._matrix.shape[0]

        data = np.full((n_trials, n_channels, n_samples), np.nan)
        for i, (_, row) in enumerate(self.epochs.iterrows()):
            lb, ub = row[['start_index', 'end_index']].astype('i')
            samples = ub-lb
            data[i, :, :samples] = self._matrix[:, lb:ub]
        return data

    def as_average_trial(self):
        '''
        Return data as a 2D array of channel x time averaged across trials
        '''
        m = self.as_trials()
        return np.nanmean(m, axis=0)

    def _get_attributes(self):
        md_attributes = ['name', 'chans', 'fs', 'epochs', 'meta',
                         'recording']
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
        nreps = len(self.epochs)
        split_rep = round(nreps*fraction)
        split_rep = np.clip(split_rep, 1, nreps-1)

        # Find the start time of the repetition
        i = self.epochs.iloc[split_rep]['start_index'].astype('i')

        data = self.as_continuous()
        ldata = data[..., :i]
        rdata = data[..., i:]

        lepochs = self.epochs.iloc[:split_rep].copy()
        repochs = self.epochs.iloc[split_rep:].copy()

        # Correct the index
        repochs[['start_index', 'end_index']] -= i

        lsignal = self._modified_copy(data=ldata, epochs=lepochs)
        rsignal = self._modified_copy(data=rdata, epochs=repochs)
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

        mask = self.epochs['end_index'] < split_idx
        lepochs = self.epochs.loc[mask]

        mask = self.epochs['start_index'] > split_idx
        repochs = self.epochs.loc[mask]
        repochs[['start_index', 'end_index']] -= split_idx

        lsignal = self._modified_copy(ldata, epochs=lepochs)
        rsignal = self._modified_copy(rdata, epochs=repochs)

        return lsignal, rsignal

    def jackknifed_by_reps(self, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with entire reps NaN'd out. If nreps is not an
        integer multiple of nsplits, an error is thrown.  Optional argument
        'invert' causes everything BUT the jackknife to be NaN.
        '''
        nreps = len(self.epochs)
        ratio = (nreps / nsplits)
        if ratio != int(ratio) or ratio < 1:
            m = 'nreps must be an integer multiple of nsplits, got {}'
            raise ValueError(m.format(ratio))

        ratio = int(ratio)
        m = self.as_trials()
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

    @classmethod
    def concatenate_time(cls, signals):
        '''
        Combines the signals along the time axis

        Requires that all signals have the same number of channels and sampling
        rate.
        '''
        # Make sure all objects passed are instances of the Signal class
        for signal in signals:
            if not isinstance(signal, Signal):
                raise ValueError('Not a signal')

        # Make sure that important attributes match up
        base = signals[0]
        for signal in signals[1:]:
            if not base.fs == signal.fs:
                raise ValueError('Cannot append signal with different fs')
            if not base.chans == signal.chans:
                raise ValueError('Cannot append signal with different channels')

        # Now, concatenate data along time axis
        data = np.concatenate([s.as_continuous() for s in signals], axis=-1)

        # Merge the trial info tables. For all signals after the first signal,
        # we need to offset the start and end indices to ensure that they
        # reflect the correct position of the trial in the merged array.
        offset = 0
        epochs = []
        for signal in signals:
            ti = signal.epochs.copy()
            ti['end_index'] += offset
            ti['start_index'] += offset
            offset += signal.ntimes
            epochs.append(ti)
        epochs = pd.concat(epochs, ignore_index=True)

        return Signal(
            name=base.name,
            recording=base.recording,
            chans=base.chans,
            fs=base.fs,
            meta=base.meta,
            matrix=data,
            epochs=epochs
        )

    @classmethod
    def concatenate_channels(cls, signals):
        '''
        Merge two signals as a set of new channels. Must have the same number of
        time samples.
        '''
        for signal in signals:
            if not isinstance(signal, Signal):
                raise ValueError('Not a signal')

        base = signals[0]
        for signal in signals[1:]:
            if not base.fs == signal.fs:
                raise ValueError('Cannot append signal with different fs')
            if not base.ntimes == signal.ntimes:
                raise ValueError('Cannot append signal with different channels')

        data = np.concatenate([s.as_continuous() for s in signals], axis=0)

        chans = []
        for signal in signals:
            chans.extend(signal.chans)

        return Signal(
            name=base.name,
            recording=base.recording,
            chans=chans,
            fs=base.fs,
            meta=base.meta,
            epochs=base.epochs,
            matrix=data,
            )

    @property
    def shape(self):
        return self._matrix.shape

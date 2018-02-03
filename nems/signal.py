import os
import copy
import json
import pandas as pd
import numpy as np

from nems.epoch import (epoch_union, epoch_difference, epoch_intersection,
                        epoch_contains, adjust_epoch_bounds, remove_overlap,
                        merge_epoch,)

class Signal:

    def __init__(self, fs, matrix, name, recording, chans=None, epochs=None,
                 meta=None):
        '''
        Parameters
        ----------
        ... TODO
        epochs : {None, DataFrame}
            Epochs are periods of time that are tagged with a name
            When defined, the DataFrame should have these first three columns:
                 ('start', 'end', 'name')
            denoting the start and end of the time of an epoch (in seconds).
            You may use the same epoch name multiple times; this is common when
            tagging epochs that correspond to repetitions of the same stimulus.
        ...
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
            # TODO: Raise some kind of warning here instead?
            #       Could be (probably rare) cases where T does end up < C
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

        if self.chans:
            if type(self.chans) is not list:
                raise ValueError('Chans must be a list.')
            typesok = [(True if type(c) is str else False) for c in self.chans]
            if not all(typesok):
                raise ValueError('Chans must be a list of strings:' +
                                 str(self.chans) + str(typesok))

        if self.fs < 0:
            m = 'Sampling rate of signal must be a positive number. Got {}.'
            raise ValueError(m.format(self.fs))

        if type(self._matrix) is not np.ndarray:
            raise ValueError('matrix must be a np.ndarray:' +
                             type(self._matrix))

    def save(self, dirpath, fmt='%.18e'):
        '''
        Save this signal to a CSV file + JSON sidecar. If desired,
        you may use optional parameter fmt (for example, fmt='%1.3e')
        to alter the precision of the floating point matrices.
        '''
        filebase = self.recording + '.' + self.name
        basepath = os.path.join(dirpath, filebase)
        csvfilepath = basepath + '.csv'
        epochfilepath = basepath + '.epoch.csv'
        jsonfilepath = basepath + '.json'

        mat = self.as_continuous()
        mat = np.swapaxes(mat, 0, 1)
        np.savetxt(csvfilepath, mat, delimiter=",", fmt=fmt)
        self.epochs.to_csv(epochfilepath, sep=',', index=False)
        with open(jsonfilepath, 'w') as fh:
            attributes = self._get_attributes()
            del attributes['epochs']
            json.dump(attributes, fh)

        return (csvfilepath, jsonfilepath)

    def copy(self):
        ''' Shorthand wrapper for copy.copy(self). '''
        return copy.copy(self)

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
        epochfilepath = basepath + '.epoch.csv'
        jsonfilepath = basepath + '.json'
        mat = pd.read_csv(csvfilepath, header=None).values
        if os.path.isfile(epochfilepath):
            epochs = pd.read_csv(epochfilepath)
        else:
            epochs = None
        mat = mat.astype('float')
        mat = np.swapaxes(mat, 0, 1)
        with open(jsonfilepath, 'r') as f:
            js = json.load(f)
            s = Signal(name=js['name'],
                       chans=js.get('chans', None),
                       epochs=epochs,
                       recording=js['recording'],
                       fs=js['fs'],
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
        Return a copy of signal data, as a 2D numpy array (channel x time).
        '''
        return self._matrix.copy()

    def _get_attributes(self):
        md_attributes = ['name', 'chans', 'fs', 'meta', 'recording', 'epochs']
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

    def split_at_time(self, fraction):
        '''
        Splits this signal at 'fraction' of the total length of the time series
        to create a tuple of two signals: (before, after).
        Example:
          l, r = mysig.split_at_time(0.8)
          assert(l.ntimes == 0.8 * mysig.ntimes)
          assert(r.ntimes == 0.2 * mysig.ntimes)
        '''
        split_idx = max(1, int(self.ntimes * fraction))
        split_time = split_idx/self.fs

        data = self.as_continuous()
        ldata = data[..., :split_idx]
        rdata = data[..., split_idx:]

        if self.epochs is None:
            lepochs = None
            repochs = None
        else:
            mask = self.epochs['start'] < split_time
            lepochs = self.epochs.loc[mask]
            mask = self.epochs['end'] > split_idx
            repochs = self.epochs.loc[mask]
            repochs[['start', 'end']] -= split_idx

        lsignal = self._modified_copy(ldata, epochs=lepochs)
        rsignal = self._modified_copy(rdata, epochs=repochs)

        return lsignal, rsignal

    def jackknifed_by_epochs(self, epoch_name, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with epochs matching epoch_name NaN'd out.
        Optional argument 'invert' causes everything BUT the matched epochs
        to be NaN'd. If no epochs are found that match the regex, an exception
        is thrown. The epochs data structure itself is not changed.
        '''
        raise NotImplementedError
        epochs = self.get_epochs(epoch_name)
        epoch_indices = (epochs * self.fs).astype('i')

        if len(epochs) == 0:
            m = 'No epochs found matching that epoch_name. Unable to jackknife.'
            raise ValueError(m)

        data = self.as_continuous()
        mask = np.zeros_like(data, dtype=np.bool)
        for lb, ub in epoch_indices:
            print(lb, ub, mask.shape)
            mask[:, lb:ub] = 1
        if invert:
            mask = ~mask
        data[mask] = np.nan
        return self._modified_copy(data)

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
        Combines the signals along the time axis. All signals must have the
        same number of channels and the same sampling rates.
        '''
        # Make sure all objects passed are instances of the Signal class
        for signal in signals:
            if not isinstance(signal, Signal):
                raise ValueError('Not a signal')

        # Make sure that important attributes match up
        base = signals[0]
        for signal in signals[1:]:
            if not base.fs == signal.fs:
                raise ValueError('Cannot concat signals with unequal fs')
            if not base.chans == signal.chans:
                raise ValueError('Cannot concat signals with unequal # of chans')

        # Now, concatenate data along time axis
        data = np.concatenate([s.as_continuous() for s in signals], axis=-1)

        # Merge the epoch tables. For all signals after the first signal,
        # we need to offset the start and end indices to ensure that they
        # reflect the correct position of the trial in the merged array.
        offset = 0
        epochs = []
        for signal in signals:
            ti = signal.epochs.copy()
            ti['end'] += offset
            ti['start'] += offset
            offset += signal.ntimes/signal.fs
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
        Given signals=[sig1, sig2, sig3, ..., sigN], concatenate all channels
        of [sig2, ...sigN] as new channels on sig1. All signals must be equal-
        length time series sampled at the same rate (i.e. ntimes and fs are the
        same for all signals).
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

        #epochs = []
        #for signal in signals:
        #    epochs.append(signal.epochs)
        epochs=signals[0].epochs

        return Signal(
            name=base.name,
            recording=base.recording,
            chans=chans,
            fs=base.fs,
            meta=base.meta,
            epochs=epochs,
            matrix=data,
            )

    def get_epoch_bounds(self, epoch, trim=False, fix_overlap=None):
        '''
        Get boundaries of named epoch.

        Parameters
        ----------
        epoch : {string, Nx2 array}
            If string, name of epoch (as stored in internal dataframe) to
            extract. If Nx2 array, the first column indicates the start time (in
            seconds) and the second column indicates the end time (in seconds)
            to extract.
        trim : boolean
            If True, ensure that epoch boundaries fall within the range of the
            signal. Epochs with boundaries falling outside the signal range will
            be truncated. For example, if an epoch runs from -1.5 to 10, it will
            be truncated to 0 to 10 (all signals start at time 0).
        fix_overlap : {None, 'merge', 'first'}
            Indicates how to handle overlapping epochs. If None, return
            boundaries as-is. If 'merge', merge overlapping epochs into a single
            epoch. If 'first', keep only the first of an overlapping set of
            epochs.

        Returns
        -------
        bounds : 2D array (n_occurances x 2)
            Each row in the array corresponds to an occurance of the epoch. The
            first column is the start time and the second column is the end
            time.
        '''
        # If string, pull the epochs out of the internal dataframe.
        if isinstance(epoch, str):
            if self.epochs is None:
                m = "Signal does not have any epochs defined"
                raise ValueError(m)
            mask = self.epochs['name'] == epoch
            bounds = self.epochs.loc[mask, ['start', 'end']].values

        if trim:
            bounds = np.clip(bounds, 0, self.ntimes*self.fs)

        if fix_overlap is None:
            pass
        elif fix_overlap == 'merge':
            bounds = merge_epoch(bounds)
        elif fix_overlap == 'first':
            bounds = remove_overlap(bounds)
        else:
            m = 'Unsupported mode, {}, for fix_overlap'.format(fix_overlap)
            raise ValueError(m)

        return bounds

    def get_epoch_indices(self, epoch, trim=False):
        '''
        Get boundaries of named epoch as index.

        Parameters
        ----------
        epoch : {string, Nx2 array}
            If string, name of epoch (as stored in internal dataframe) to
            extract. If Nx2 array, the first column indicates the start time (in
            seconds) and the second column indicates the end time (in seconds)
            to extract.
        trim : boolean
            If True, ensure that epoch boundaries fall within the range of the
            signal. Epochs with boundaries falling outside the signal range will
            be truncated. For example, if an epoch runs from -1.5 to 10, it will
            be truncated to 0 to 10 (all signals start at time 0).
        fix_overlap : {None, 'merge', 'first'}
            Indicates how to handle overlapping epochs. If None, return
            boundaries as-is. If 'merge', merge overlapping epochs into a single
            epoch. If 'first', keep only the first of an overlapping set of
            epochs.

        Returns
        -------
        bounds : 2D array (n_occurances x 2)
            Each row in the array corresponds to an occurance of the epoch. The
            first column is the start time and the second column is the end
            time.
        '''
        bounds = self.get_epoch_bounds(epoch, trim)
        return (bounds * self.fs).astype('i')

    def extract_epoch(self, epoch):
        '''
        Extracts all occurances of epoch from the signal.

        Parameters
        ----------
        epoch : {string, Nx2 array}
            If string, name of epoch (as stored in internal dataframe) to
            extract. If Nx2 array, the first column indicates the start time (in
            seconds) and the second column indicates the end time (in seconds)
            to extract.

        Returns
        -------
        epoch_data : 3D array
            Three dimensional array of shape O, C, T where O is the number of
            occurances of the epoch, C is the number of channels, and T is the
            maximum length of the epoch in samples.

        Note
        ----
        Epochs tagged with the same name may have various lengths. Shorter
        epochs will be padded with NaN.
        '''
        epoch_indices = self.get_epoch_indices(epoch, trim=True)
        if epoch_indices.size == 0:
            raise IndexError("No matching epochs to extract for: {}"
                             .format(epoch))
        n_samples = np.max(epoch_indices[:, 1]-epoch_indices[:, 0])
        n_epochs = len(epoch_indices)

        epoch_data = np.full((n_epochs, self.nchans, n_samples), np.nan)
        for i, (lb, ub) in enumerate(epoch_indices):
            samples = ub-lb
            epoch_data[i, :, :samples] = self._matrix[:, lb:ub]

        return epoch_data

    def average_epoch(self, epoch):
        '''
        Returns the average of the epoch.

        Parameters
        ----------
        epoch : {string, Nx2 array}
            If string, name of epoch (as stored in internal dataframe) to
            extract. If Nx2 array, the first column indicates the start time (in
            seconds) and the second column indicates the end time (in seconds)
            to extract.

        Returns
        -------
        mean_epoch : 2D array
            Two dimensinonal array of shape C, T where C is the number of
            channels, and T is the maximum length of the epoch in samples.
        '''
        epoch_data = self.extract_epoch(epoch)
        return np.nanmean(epoch_data, axis=0)

    def extract_epochs(self, epoch_names):
        '''
        Returns a dictionary of the data matching each element in epoch_names.

        Parameters
        ----------
        epoch_names : list
            List of epoch names to extract. These will be keys in the result
            dictionary.

        Returns
        -------
        epoch_datasets : dict
            Keys are the names of the epochs, values are 3D arrays created by
            `extract_epoch`.
        '''
        # TODO: Update this to work with a mapping of key -> Nx2 epoch
        # structure as well.
        return {name: self.extract_epoch(name) for name in epoch_names}

    def replace_epoch(self, epoch, epoch_data):
        '''
        Returns a new signal, created by replacing every occurrence of
        epoch with epoch_data, assumed to be a 2D matrix of data
        (chans x time).
        '''
        data = self.as_continuous()
        indices = self.get_epoch_indices(epoch)
        if indices.size == 0:
            raise RuntimeWarning("No occurences of epoch were found: \n{}\n"
                                 "Nothing to replace.".format(epoch))
        for lb, ub in indices:
            data[:, lb:ub] = epoch_data

        return self._modified_copy(data)

    def replace_epochs(self, epoch_dict):
        '''
        Returns a new signal, created by replacing every occurrence of epochs
        in this signal with whatever is found in the replacement_dict under
        the same epoch_name key. Dict values are assumed to be 2D matrices.

        If the replacement matrix shape is not the same as the original
        epoch being replaced, an exception will be thrown.

        If overlapping epochs are defined, then they will be replaced in
        the order present in the epochs dataframe (i.e. sorting your
        epochs dataframe may change the results you get!). For this reason,
        we do not recommend replacing overlapping epochs in a single
        operation because there is some ambiguity as to the result.
        '''
        # TODO: Update this to work with a mapping of key -> Nx2 epoch
        # structure as well.
        data = self.as_continuous()
        for epoch, epoch_data in epoch_dict.items():
            for lb, ub in self.get_epoch_indices(epoch):

                # SVD kludge to deal with rounding from floating-point time
                # to integer bin index
                if ub-lb < epoch_data.shape[1]:
                    ub += epoch_data.shape[1]-(ub-lb)

                data[:, lb:ub] = epoch_data

        return self._modified_copy(data)

    def epoch_to_signal(self, epoch_name):
        '''
        Convert an epoch to a signal using the same sampling rate and duration
        as this signal.

        Parameters
        ----------
        epoch_name : string
            Epoch to convert to a signal

        Returns
        -------
        signal : instance of Signal
            A signal whose value is 1 for each occurence of the epoch, 0
            otherwise.
        '''
        data = np.zeros([1,self.ntimes], dtype=np.bool)
        for lb, ub in self.get_epoch_indices(epoch_name, trim=True):
            data[:, lb:ub] = 1
        return self._modified_copy(data, chans=[epoch_name])

    def select_epoch(self, epoch):
        '''
        Returns a new signal, the same as this, with everything NaN'd
        unless it is tagged with epoch_name.
        '''
        new_data = np.full(self.shape, np.nan)
        for (lb, ub) in self.get_epoch_indices(epoch, trim=True):
            new_data[:, lb:ub] = self._matrix[:, lb:ub]
        if np.all(np.isnan(new_data)):
            raise RuntimeWarning("No matched occurrences for epoch: \n{}\n"
                                 "Returned signal will be only NaN."
                                 .format(epoch))
        return self._modified_copy(new_data)

    def select_epochs(self, list_of_epoch_names):
        '''
        Returns a new signal, the same as this, with everything NaN'd
        unless it is tagged with one of the epoch_names found in
        list_of_epoch_names.
        '''
        # TODO: Update this to work with a mapping of key -> Nx2 epoch
        # structure as well.
        new_data = np.full(self.shape, np.nan)
        for epoch_name in list_of_epoch_names:
            for (lb, ub) in self.get_epoch_indices(epoch_name, trim=True):
                new_data[:, lb:ub] = self._matrix[:, lb:ub]
        if np.all(np.isnan(new_data)):
            raise RuntimeWarning("No matched occurrences for epochs: \n{}\n"
                                 "Returned signal will be only NaN."
                                 .format(list_of_epoch_names))
        return self._modified_copy(new_data)

    def trial_epochs_from_reps(self, nreps=1):
        """
        Creates a generic epochs DataFrame with a number of trials based on
        sample length and number of repetitions specified.

        Example
        -------
        If signal._matrix has shape 3x100 and the signal is sampled at 100 Hz,
        trial_epochs_from_reps(nreps=5) would generate a DataFrame with 5 trials
        (starting at 0, 0.2, 0.4, 0.6, 0.8 seconds).

        Note
        ----
        * The number of time samples must be evenly divisible by the number of
          repetitions.
        * Epoch indices behave similar to python list indices, so start is
          inclusive while end is exclusive.
        """

        trial_size = self.ntimes/nreps/self.fs
        if self.ntimes % nreps:
            m = 'Signal not evenly divisible into fixed-length trials'
            raise ValueError(m)

        starts = np.arange(nreps) * trial_size
        ends = starts + trial_size
        return pd.DataFrame({
            'start': starts,
            'end': ends,
            'name': 'trial'
        })

        '''
        Returns a copy of a view of just the epochs matching epoch_name.
        If no matching epochs are found, returns None.
        '''
        idxs = self.epochs[self.epochs['epoch_name'] == epoch_name]

        if any(idxs):
            return idxs.copy()
        else:
            return None

    def add_epoch(self, epoch_name, epoch):
        '''
        Add epoch to the internal epochs dataframe

        Parameters
        ----------
        epoch_name : string
            Name of epoch
        epoch : 2D array of (M x 2)
            The first column is the start time and second column is the end
            time. M is the number of occurences of the epoch.
        '''
        df = pd.DataFrame(epoch, columns=['start', 'end'])
        df['name'] = epoch_name
        if self.epochs is not None:
            self.epochs = self.epochs.append(df, ignore_index=True)
        else:
            self.epochs = df

    @property
    def shape(self):
        return self._matrix.shape



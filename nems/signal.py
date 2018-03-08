import io
import os
import json
import re
import math
import pandas as pd
import numpy as np
import tempfile
import copy
from nems.epoch import remove_overlap, merge_epoch, verify_epoch_integrity

class Signal:

    def __init__(self, fs, matrix, name, recording, chans=None, epochs=None,
                 meta=None, safety_checks=True):
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
            tagging epochs that correspond to occurrences of the same stimulus.
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
        if safety_checks and T < C:
            m = 'Incorrect matrix dimensions?: (C, T) is {}. ' \
                'We expect a long time series, but T < C'
            raise RuntimeWarning(m.format((C, T)))

        self.nchans = C
        self.ntimes = T

        # Cached properties for speed; their use is however optional
        # TODO: Can these be made lazy? There is actually a big performance hit in precalculating these
        #self.channel_max = np.nanmax(self._matrix, axis=-1, keepdims=True)
        #self.channel_min = np.nanmin(self._matrix, axis=-1, keepdims=True)
        #self.channel_mean = np.nanmean(self._matrix, axis=-1, keepdims=True)
        #self.channel_var = np.nanvar(self._matrix, axis=-1, keepdims=True)
        #self.channel_std = np.nanstd(self._matrix, axis=-1, keepdims=True)

        if safety_checks and not isinstance(self.name, str):
            m = 'Name of signal must be a string: {}'.format(self.name)
            raise ValueError(m)

        if safety_checks and not isinstance(self.recording, str):
            m = 'Name of recording must be a string: {}'.format(self.recording)
            raise ValueError(m)

        if safety_checks and self.chans:
            if type(self.chans) is not list:
                raise ValueError('Chans must be a list.')
            typesok = [(True if type(c) is str else False) for c in self.chans]
            if not all(typesok):
                raise ValueError('Chans must be a list of strings:' +
                                 str(self.chans) + str(typesok))
            # Test that channel names use only lowercase letters and numbers 0-9
            for s in self.chans:
                if s and not self._string_syntax_valid(s):
                    raise ValueError("Disallowed characters in: {0}\n"
                                     .format(s))

        # Test that other names use only lowercase letters and numbers 0-9
        if safety_checks:
            for s in [name, recording]:
                if s and not self._string_syntax_valid(s):
                    raise ValueError("Disallowed characters in: {0}\n"
                                     .format(s))

        if safety_checks and self.fs < 0:
            m = 'Sampling rate of signal must be a positive number. Got {}.'
            raise ValueError(m.format(self.fs))

        if safety_checks and type(self._matrix) is not np.ndarray:
            raise ValueError('matrix must be a np.ndarray:' +
                             type(self._matrix))

        # not implemented yet in epoch.py -- 2/4/2018f
        # verify_epoch_integrity(self.epochs)

    def as_file_streams(self, fmt='%.18e'):
        '''
        Returns 3 filestreams for this signal: the csv, json, and epoch.
        TODO: Better docs and a refactoring of this and save()
        '''
        # TODO: actually compute these instead of cheating with a tempfile
        files = {}
        filebase = self.recording + '.' + self.name
        csvfile = filebase + '.csv'
        jsonfile = filebase + '.json'
        epochfile = filebase + '.epoch.csv'
        # Create three streams
        files[csvfile] = io.BytesIO()
        files[jsonfile] = io.StringIO()
        files[epochfile] = io.StringIO()
        # Write to those streams
        # Write the CSV file to a bytesIO buffer
        mat = self.as_continuous()
        mat = np.swapaxes(mat, 0, 1)
        np.savetxt(files[csvfile], mat, delimiter=",", fmt=fmt)
        files[csvfile].seek(0)  # Seek back to start of file
        # TODO: make epochs optional?
        self.epochs.to_csv(files[epochfile], sep=',', index=False)
        # Write the JSON stream
        attributes = self._get_attributes()
        del attributes['epochs']
        json.dump(attributes, files[jsonfile])
        return files

    def save(self, dirpath, fmt='%.18e'):
        '''
        Save this signal to a CSV file + JSON sidecar. If desired,
        you may use optional parameter fmt (for example, fmt='%1.3e')
        to alter the precision of the floating point matrices.
        '''
        filebase = self.recording + '.' + self.name
        basepath = os.path.join(dirpath, filebase)
        csvfilepath = basepath + '.csv'
        jsonfilepath = basepath + '.json'
        epochfilepath = basepath + '.epoch.csv'

        mat = self.as_continuous()
        mat = np.swapaxes(mat, 0, 1)
        # TODO: Why does numpy not support fileobjs like streams?
        np.savetxt(csvfilepath, mat, delimiter=",", fmt=fmt)
        self.epochs.to_csv(epochfilepath, sep=',', index=False)
        with open(jsonfilepath, 'w') as fh:
            attributes = self._get_attributes()
            del attributes['epochs']
            json.dump(attributes, fh)

        return (csvfilepath, jsonfilepath, epochfilepath)

    @staticmethod
    def load(basepath):
        '''
        Loads the CSV & JSON files at basepath; returns a Signal() object.
        Example: If you want to load
           /tmp/sigs/gus027b13_p_PPS_resp-a1.csv
           /tmp/sigs/gus027b13_p_PPS_resp-a1.json
        then give this function
           /tmp/sigs/gus027b13_p_PPS_resp-a1
        '''
        csvfilepath = basepath + '.csv'
        epochfilepath = basepath + '.epoch.csv'
        jsonfilepath = basepath + '.json'
        # TODO: reduce code duplication and call load_from_streams
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
    def load_from_streams(csv_stream, json_stream, epoch_stream=None):
        ''' Loads from BytesIO objects rather than files. '''
        # Read the epochs stream if it exists
        epochs = pd.read_csv(epoch_stream) if epoch_stream else None
        # Read the json metadata
        js = json.load(json_stream)
        # Read the CSV
        mat = pd.read_csv(csv_stream, header=None).values
        mat = mat.astype('float')
        mat = np.swapaxes(mat, 0, 1)
        # mat = np.genfromtxt(csv_stream, delimiter=',')
        # Now build the signal
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
        Returns a list of all CSV/JSON pairs files found in DIRECTORY,
        Paths are relative, not absolute.
        '''
        files = os.listdir(directory)
        return Signal._csv_and_json_pairs(files)

    @staticmethod
    def _csv_and_json_pairs(files):
        '''
        Given a list of files, return the file basenames (i.e. no extensions)
        that for which a .CSV and a .JSON file exists.
        '''
        just_fileroot = lambda f: os.path.splitext(os.path.basename(f))[0]
        csvs = [just_fileroot(f) for f in files if f.endswith('.csv')]
        jsons = [just_fileroot(f) for f in files if f.endswith('.json')]
        overlap = set.intersection(set(csvs), set(jsons))
        return list(overlap)

    @staticmethod
    def _string_syntax_valid(s):
        '''
        Returns True iff the string is valid for use in signal names,
        recording names, or channel names. Else False.
        '''
        disallowed = re.compile('[^a-zA-Z0-9_\-]')
        match = disallowed.findall(s)
        if match:
            return False
        else:
            return True

    def as_continuous(self):
        '''
        Return a copy of signal data, as a 2D numpy array (channel x time).
        '''
        return self._matrix.copy()

    def _get_attributes(self):
        md_attributes = ['name', 'chans', 'fs', 'meta', 'recording', 'epochs']
        return {name: getattr(self, name) for name in md_attributes}

    def copy(self):
        '''
        Returns a copy of this signal.
        '''
        return copy.copy(self)
    
    def _modified_copy(self, data, **kwargs):
        '''
        For internal use when making various immutable copies of this signal.
        '''
        attributes = self._get_attributes()
        attributes.update(kwargs)
        return Signal(matrix=data, safety_checks=False, **attributes)

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
            mask = self.epochs['end'] > split_time
            repochs = self.epochs.loc[mask]
            repochs[['start', 'end']] -= split_time

        # If epochs were present initially but missing after split,
        # raise a warning.
        portion = None
        if lepochs.size == 0:
            portion = 'first'
        elif repochs.size == 0:
            portion = 'second'
        if portion:
            raise RuntimeWarning("Epochs for {0} portion of signal: {1}"
                                 "ended up empty after splitting by time."
                                 .format(portion, self.name))

        lsignal = self._modified_copy(ldata, epochs=lepochs)
        rsignal = self._modified_copy(rdata, epochs=repochs)

        return lsignal, rsignal

    def split_by_epochs(self, epochs_for_est, epochs_for_val):
        '''
        Returns a tuple of estimation and validation data splits: (est, val).
        Arguments should be lists of epochs that define the estimation and
        validation sets. Both est and val will have non-matching data NaN'd out.
        '''
        est = self.select_epochs(epochs_for_est)
        val = self.select_epochs(epochs_for_val)
        return (est, val)

    def jackknife_by_epoch(self, njacks, jack_idx, epoch_name,
                           tiled=True,
                           invert=False):
        '''
        Returns a new signal, with epochs matching epoch_name NaN'd out.
        Optional argument 'invert' causes everything BUT the matched epochs
        to be NaN'd. njacks determines the number of jackknifes to divide
        the epochs into, and jack_idx determines which one to return.

        'Tiled' makes each jackknife use every njacks'th occurrence, and is
        probably best explained by the following example...

        If there are 18 occurrences of an epoch, njacks=5, invert=False,
        and tiled=True, then the five jackknifes will have these
        epochs NaN'd out:

           jacknife[0]:  0, 5, 10, 15
           jacknife[1]:  1, 6, 11, 16
           jacknife[2]:  2, 7, 12, 17
           jacknife[3]:  3, 8, 13
           jacknife[4]:  4, 9, 14

        Note that the last two jackknifes have one fewer occurrences.

        If tiled=False, then the pattern of NaN'd epochs becomes sequential:

           jacknife[0]:   0,  1,  2,  3
           jacknife[1]:   4,  5,  6,  7,
           jacknife[2]:   8,  9, 10, 11,
           jacknife[3]:  12, 13, 14, 15,
           jacknife[4]:  16, 17

        Here we can see the last jackknife has 2 fewer occurrences.

        In any case, an exception will be thrown if epoch_name is not found,
        or when there are fewer occurrences than njacks.
        '''

        epochs = self.get_epoch_bounds(epoch_name)
        epoch_indices = (epochs * self.fs).astype('i').tolist()
        occurrences, _ = epochs.shape

        if len(epochs) == 0:
            m = 'No epochs found matching that epoch_name. Unable to jackknife.'
            raise ValueError(m)

        if occurrences < njacks:
            raise ValueError("Can't divide {0} occurrences into {1} jackknifes"
                             .format(occurrences, njacks))

        if jack_idx < 0 or njacks < 0:
            raise ValueError("Neither jack_idx nor njacks may be negative")

        nrows = math.ceil(occurrences / njacks)
        idx_matrix = np.arange(nrows * njacks)

        if tiled:
            idx_matrix = idx_matrix.reshape(nrows, njacks)
            idx_matrix = np.swapaxes(idx_matrix, 0, 1)
        else:
            idx_matrix = idx_matrix.reshape(njacks, nrows)

        data = self.as_continuous()
        mask = np.zeros_like(data, dtype=np.bool)

        for idx in idx_matrix[jack_idx].tolist():
            if idx < occurrences:
                lb, ub = epoch_indices[idx]
                mask[:, lb:ub] = 1

        if invert:
            mask = ~mask

        data[mask] = np.nan

        return self._modified_copy(data)

    def jackknifes_by_epoch(self, njacks, epoch_name, tiled=True):
        '''
        Convenience fn. Returns generator that returns njacks tuples of
        (est, val) made using jackknife_by_epoch().
        '''
        jack_idx = 0
        while jack_idx < njacks:
            yield (self.jackknife_by_epoch(njacks, jack_idx, epoch_name,
                                           tiled=tiled, invert=False),
                   self.jackknife_by_epoch(njacks, jack_idx, epoch_name,
                                           tiled=tiled, invert=True))
            jack_idx += 1

    def jackknife_by_time(self, njacks, jack_idx, invert=False, excise=False):
        '''
        Returns a new signal, with some data NaN'd out based on its position
        in the time stream. jack_idx is indexed from 0; if you have 20 splits,
        the first is #0 and the last is #19.
        Optional argument 'invert' causes everything BUT the jackknife to be NaN.
        Optional argument 'excise' removes the elements that would've been NaN
        and thus changes the size of the signal.
        '''
        splitsize = int(self.ntimes / njacks)
        if splitsize < 1:
            m = 'Too many jackknifes? Splitsize was {}'
            raise ValueError(m.format(splitsize))

        split_start = jack_idx * splitsize
        if jack_idx == njacks - 1:
            split_end = self.ntimes
        else:
            split_end = (jack_idx + 1) * splitsize

        m = self.as_continuous()
        if excise:
            if invert:
                o = np.empty((self.nchans, split_end - split_start))
                o = m[:, split_start:split_end]
            else:
                o = np.delete(m, slice(split_start, split_end), axis=-1)
            return self._modified_copy(o.reshape(self.nchans, -1))
        else:
            if not invert:
                m[..., split_start:split_end] = np.nan
            else:
                mask = np.ones_like(m, dtype=np.bool)
                mask[:, split_start:split_end] = 0
                m[mask] = np.nan
            return self._modified_copy(m.reshape(self.nchans, -1))

    def jackknifes_by_time(self, njacks):
        '''
        Convenience fn. Returns generator that returns njacks tuples of
        (est, val) made using jackknife_by_time().
        '''
        jack_idx = 0
        while jack_idx < njacks:
            yield (self.jackknife_by_time(njacks, jack_idx, invert=False),
                   self.jackknife_by_time(njacks, jack_idx, invert=True))
            jack_idx += 1

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
            epochs=epochs,
            safety_checks=False
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
            if signal.chans:
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
            safety_checks=False
            )

    def extract_channels(self, chans):
        '''
        Returns a new signal object containing only the specified
        channel indices.
        '''
        array = self.as_continuous()
        if isinstance(chans, int):
            chans = [chans]
        c,t = self.shape
        removals = []
        for i in range(c):
            if i not in chans:
                removals.append(i)
        new_array = np.delete(array, removals, axis=0)
        return self._modified_copy(new_array)

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
            raise IndexError("No matching epochs to extract for: {0}\n"
                             "In signal: {1}"
                             .format(epoch, self.name))
        n_samples = np.max(epoch_indices[:, 1]-epoch_indices[:, 0])
        n_epochs = len(epoch_indices)

        epoch_data = np.full((n_epochs, self.nchans, n_samples), np.nan)
        for i, (lb, ub) in enumerate(epoch_indices):
            samples = ub-lb
            epoch_data[i, :, :samples] = self._matrix[:, lb:ub]

        return epoch_data

    def count_epoch(self, epoch):
        """Returns the number of occurrences of the given epoch."""
        epoch_indices = self.get_epoch_indices(epoch, trim=True)
        count = len(epoch_indices)
        return count

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
            raise RuntimeWarning("No occurrences of epoch were found: \n{}\n"
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
                    # epoch data may be too long bc padded with nans,
                    # truncate!
                    epoch_data=epoch_data[:,0:(ub-lb)]
                    #ub += epoch_data.shape[1]-(ub-lb)
                elif ub-lb > epoch_data.shape[1]:
                    ub -= (ub-lb)-epoch_data.shape[1]
                if ub>data.shape[1]:
                    ub -= ub-data.shape[1]
                    epoch_data=epoch_data[:,0:(ub-lb)]
                #print("Data len {0} {1}-{2} {3}".format(
                #        data.shape[1],lb,ub,ub-lb))
                #print(data[:, lb:ub].shape)
                #print(epoch_data.shape)
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
            A signal whose value is 1 for each occurrence of the epoch, 0
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

    def trial_epochs_from_occurrences(self, occurrences=1):
        """
        Creates a generic epochs DataFrame with a number of trials based on
        sample length and number of occurrences specified.

        Example
        -------
        If signal._matrix has shape 3x100 and the signal is sampled at 100 Hz,
        trial_epochs_from_occurrences(occurrences=5) would generate a DataFrame
        with 5 trials (starting at 0, 0.2, 0.4, 0.6, 0.8 seconds).

        Note
        ----
        * The number of time samples must be evenly divisible by the number of
          occurrences.
        * Epoch indices behave similar to python list indices, so start is
          inclusive while end is exclusive.
        """

        trial_size = self.ntimes/occurrences/self.fs
        if self.ntimes % occurrences:
            m = 'Signal not evenly divisible into fixed-length trials'
            raise ValueError(m)

        starts = np.arange(occurrences) * trial_size
        ends = starts + trial_size
        return pd.DataFrame({
            'start': starts,
            'end': ends,
            'name': 'trial'
        })

    def add_epoch(self, epoch_name, epoch):
        '''
        Add epoch to the internal epochs dataframe

        Parameters
        ----------
        epoch_name : string
            Name of epoch
        epoch : 2D array of (M x 2)
            The first column is the start time and second column is the end
            time. M is the number of occurrences of the epoch.
        '''
        df = pd.DataFrame(epoch, columns=['start', 'end'])
        df['name'] = epoch_name
        if self.epochs is not None:
            self.epochs = self.epochs.append(df, ignore_index=True)
        else:
            self.epochs = df

    def transform(self, fn, newname=None):
        '''
        Applies this signal's 2d .as_continuous() matrix representation to
        function fn, which must be a pure (curried) function of one argument.

        It then packs the return value of fn into a new signal object,
        identical to this one but with different data.

        Optional argument newname allows a new signal name to be returned.
        '''
        # x = self.as_continuous()   # Always Safe but makes a copy
        x = self._matrix  # Much faster; TODO: Test if throws warnings
        y = fn(x)
        newsig = self._modified_copy(y)
        if newname:
            newsig.name = newname
        return newsig

    def nan_outliers(self, trim_outside_zscore=2.0):
        '''
        Tries to NaN out outliers from the signal. Outliers are defined
        as being values further than trim_outside_zscore stddevs from the mean.

        Arguments:
        ----------
        trim_outside_zscore: float
        Multiple of standard deviation that determines the range of
        'normal' versus 'outlier' values.

        Returns:
        --------
        A new copy of the signal with outliers NaN'd out.
        '''
        m = self.as_continuous()
        std = np.std(m)
        mean = np.mean(m)
        max_val = mean + std * trim_outside_zscore
        min_val = mean - std * trim_outside_zscore
        m[m < max_val] = np.nan
        m[m > min_val] = np.nan
        return self._modified_copy(m)

    @property
    def shape(self):
        return self._matrix.shape


# -----------------------------------------------------------------------------
# Functions that work on multiple signal objects

def merge_selections(signals):
    '''
    Returns a new signal object by combining a list of signals of the same
    shape that are assumed to be non-overlapping selections. The returned
    signal will have identical metadata to the first signal in the list.

    For signals to be non-overlapping, every corresponding element of
    every signal must be NaN unless it is NaN in all other signals, or
    it is identical to all other values that are non-NaN.

    Ex:
    s1: [   1,   2,   3, NaN, NaN, NaN]
    s2: [ NaN, NaN, NaN,   4,   5,   6]

    would merge to:
    s3: [   1,   2,   3,   4,   5,   6]

    But this would result in error:
    s4: [   1,   2,   3,   4, NaN, NaN]
    s5: [ NaN, NaN, NaN,   4,   5,   6]

    Because the index position of  4 was non-NaN in more than one signal.
    '''

    # Check that all signals have the same shape, fs, and chans
    for s in signals:
        if s.shape != signals[0].shape:
            raise ValueError("All signals must have the same shape.")
        if s.fs != signals[0].fs:
            raise ValueError("All signals must have the same fs.")
        if s.chans != signals[0].chans:
            raise ValueError("All signals must have the same chans.")

    # Make a big 3D array from the 2D arrays
    arys = [s.as_continuous() for s in signals]
    bigary = np.stack(arys, axis=2)

    # If there are no overlapping values, then nanmean() will be equal
    # to the value found in each position
    the_mean = np.nanmean(bigary, axis=2)
    for a in arys:
        if not np.array_equal(a[np.isfinite(a)],
                              the_mean[np.isfinite(a)]):
            raise ValueError("Overlapping, unequal non-NaN values found.")

    # Use the first signal as a template for setting fs, chans, etc.
    return signals[0]._modified_copy(the_mean)

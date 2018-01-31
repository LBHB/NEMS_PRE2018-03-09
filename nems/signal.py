import os
import copy
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
            Epochs are periods of time that are tagged with a name
            When defined, the DataFrame should have these first three columns:
                 ('start_time', 'end_index', 'epoch_name')
            denoting the start and end of the time of an epoch, and what
            it is named. You may reuse the same name (because several epochs
            might correspond to the same stimulus, for example). start_time
            is inclusive, and end_index is not, like other indexing in python.

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
            raise ValueError('matrix must be a np.ndarray:'
                             + type(self._matrix))

    def save(self, dirpath, fmt='%.18e'):
        '''
        Save this signal to a CSV + JSON sidecar. If desired, you may
        use optional parameter fmt (for example, fmt='%1.3e')
        to alter the precision of the floating point matrices.
        '''
        filebase = self.recording + '.' + self.name
        basepath = os.path.join(dirpath, filebase)
        csvfilepath = basepath + '.csv'
        epochfilepath = basepath + 'epoch.csv'
        jsonfilepath = basepath + '.json'

        mat = self.as_continuous()
        mat = np.swapaxes(mat, 0, 1)
        np.savetxt(csvfilepath, mat, delimiter=",", fmt=fmt)
        # TODO:
        with open(jsonfilepath, 'w') as fh:
            attributes = self._get_attributes()
            del attributes['epochs']
            json.dump(attributes, fh)

        return (csvfilepath, jsonfilepath)

    def copy(self):
        """Wrapper for copy.copy(self)."""
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
            epochs = pd.read_csv(epochfilepath, header=None).values
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
        Return data as a 2D array of channel x time
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


    def get_epochs(self, epoch_name):

        mask = self.epochs['name'] == epoch_name
        return self.epochs.loc[mask]

    def jackknifed_by_epochs(self, epoch_name, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with epochs matching  NaN'd out.
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
        same number of channels and sampling rates.
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

        epochs = []
        for signal in signals:
            epochs.append(signal.epochs)

        return Signal(
            name=base.name,
            recording=base.recording,
            chans=chans,
            fs=base.fs,
            meta=base.meta,
            epochs=epochs,
            matrix=data,
            )

    # TODO: classmethod?
    # TODO: Have a flag 'allow_data_duplication=True' or False
    # that NaNs out data if it was already used in another epoch
    def extract_epochs(self, epoch_name):
        """
        Returns matrix with (O, C, T) where:
            O   is the number of occurences of epoch_name in the signal
            C   is the number of channels
            T   is the number of samples in time

        Because epochs tagged with the same epoch_name may have various
        lengths, for epochs with uneven length, NaNs will be appended
        to the shorter lengths to fill out the matrix.

        Example: Given that signal.nchans == 3 and that

           signal.epochs = {'start_index': [0, 10, 15],
                            'end_index': [10, 15, 35],
                            'epoch_name': [trial, trial, trial]}

        then this will be true:

           assert(signal.extract_epochs('trial').shape == (3, 3, 20))

        i.e. 3 epochs x 3 channels x 20 time samples (longest). The three
        epochs would contain 10, 15, and 0 NaN values, respectively.
        """
        if self.epochs is None:
            m = "Signal.epochs must be defined in order to fold by epochs"
            raise ValueError(m)

        epochs = self.get_epochs(epoch_name)
        epoch_indices = epochs[['start', 'end']] * self.fs
        epoch_indices = epoch_indices.astype('i')
        n_samples = np.max(epoch_indices['end']-epoch_indices['start'])
        n_epochs = len(epoch_indices)

        epoch_data = np.full((n_epochs, self.nchans, n_samples), np.nan)
        for i, (_, row) in enumerate(epoch_indices.iterrows()):
            lb = np.clip(row['start'], 0, self.ntimes)
            ub = np.clip(row['end'], 0, self.ntimes)
            samples = ub-lb
            epoch_data[i, :, :samples] = self._matrix[:, lb:ub]

        return epoch_data


    def average_epoch(self, epoch_name):
        epoch_data = self.extract_epochs(epoch_name)
        return np.nanmean(epoch_data, axis=0)

    def trial_epochs_from_reps(self, nreps=1):
        """
        Creates a generic epochs DataFrame with a number of trials
        based on sample length and number of repetitions specified.

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

    def just_epochs_named(self, epoch_name):
        '''
        Returns a copy of a view of just the epochs matching epoch_name.
        If no matching epochs are found, returns None.
        '''
        idxs = self.epochs[self.epochs['epoch_name'] == epoch_name]

        if any(idxs):
            return idxs.copy()
        else:
            return None

    def add_epochs(self, epoch_name, epoch_dataframe):
        '''
        Adds the epoch_times to this signal's epochs data structure
        under epoch_name. Will not add epochs if another epoch
        of the same name already exists.
        '''
        if type(epoch_dataframe) is not pd.DataFrame:
            raise TypeError('epoch_times must be a dataframe')

        mask = self.epochs['epoch_name'] == epoch_name
        existing_matches = self.epochs[mask]

        if not existing_matches.empty:
            raise ValueError('Epochs named that already exist!')

        epoch_dataframe['epoch_name'] = epoch_name
        self.epochs.append(epoch_dataframe)

    def extend_epoch(self, epoch_name, prepend, postpend):
        '''
        Subtract prepend from the start_time of every epoch named
        'epoch_name', add postpend from the end_time, and return
        a new dataframe containing the new epochs.

        This does not alter self.epochs -- you must do that yourself:
        # Create epochs starting 200 samples before every blink
        preblink_epochs = sig.resize_epochs('blink', 200, 0)
        print(preblink_epochs)
        sig.add_epochs(preblink_epochs)
        '''
        ep = self.just_epochs_named(epoch_name)
        ep['start_index'] -= prepend
        ep['end_index'] += postpend
        ep = ep.drop('epoch_name', 1)  # Was: ep['epoch_name'] = None
        return ep

    @staticmethod
    def indexes_of_trues(boolean_array):
        '''
        Returns the a list of start, end indexes of every contiguous block
        of True elements in numpy boolean_array. Just a helper function.
        '''
        assert(type(boolean_array) is np.ndarray)

        idxs = np.where(np.diff(boolean_array))[0].tolist()

        # Special case: first element is true
        if boolean_array[0]:
            idxs.insert(0, -1)

        # Special case: last element is true
        if boolean_array[-1]:
            idxs.append(len(boolean_array)-1)

        indexes = [[idxs[i]+1, idxs[i+1]+1] for i in range(0, len(idxs) - 1, 2)]

        return indexes

    def combine_epochs(self, name1, name2, op=None):
        '''
        Returns a new epoch based on the combination of two other epochs.
        Operator may be 'union', 'intersection', or 'difference', which
        correspond to the set operation performed on the epochs.

        Note that 'difference' is not commutative, and that
        (name1 - name2) is not equal to (name2 - name1).
        '''

        if type(name1) is pd.DataFrame:
            ep1 = name1
        else:
            ep1 = self.just_epochs_named(name1)

        if type(name2) is pd.DataFrame:
            ep2 = name2
        else:
            ep2 = self.just_epochs_named(name2)

        # TODO: Rewrite this so that it does not use temporary
        # boolean mask arrays
        overall_start = min(ep1['start_index'].min(),
                            ep2['start_index'].min())
        overall_end = max(ep1['end_index'].max(),
                          ep2['end_index'].max())
        length = overall_end - overall_start
        mask1 = np.full((length, 1), False)
        mask2 = np.full((length, 1), False)

        # Fill the boolean masks with True values where appropriate
        for e1 in ep1.values.tolist():
            start, end, name = e1
            s = max(0, start - overall_start)
            e = min(max(0, end - overall_start), length)
            mask1[s:e] = True

        for e2 in ep2.values.tolist():
            start, end, name = e2
            s = max(0, start - overall_start)
            e = min(max(0, end - overall_start), length)
            mask2[s:e] = True

        # Now do the boolean operation
        if op is 'union':
            mask = np.logical_or(mask1, mask2)
        elif op is 'intersection':
            mask = np.logical_and(mask1, mask2)
        elif op is 'difference':
            # mask = np.logical_xor(mask1, mask2)
            mask = np.logical_xor(mask1, np.logical_and(mask1, mask2))
            # mask = np.logical_and(masktmp, mask1)
        else:
            raise ValueError('operator was invalid')

        # Convert the boolan mask back into a dataframe
        idxs = self.indexes_of_trues(mask.flatten())
        starts = [i[0] + overall_start for i in idxs]
        ends = [i[1] + overall_start for i in idxs]
        new_epoch = pd.DataFrame({'start_index': starts,
                                  'end_index': ends},
                                 columns=['start_index',
                                          'end_index',
                                          'epoch_name'])

        return new_epoch

    def overlapping_epochs(self, epoch_name1, epoch_name2):
        '''
        Return the outermost boundaries of whenever epoch_name1 and
        both occured and overlapped one another.
        '''

        if type(epoch_name1) is pd.DataFrame:
            ep1 = epoch_name1
        else:
            ep1 = self.just_epochs_named(epoch_name1)

        if type(epoch_name2) is pd.DataFrame:
            ep2 = epoch_name2
        else:
            ep2 = self.just_epochs_named(epoch_name2)

        # TODO: Replace this N^2 algorithm with somthing more efficient
        pairs = []
        for (e1start, e1end, _) in ep1.values.tolist():
            for (e2start, e2end, _) in ep2.values.tolist():
                if e1start <= e2start and e1end >= e2end:
                    # E2 occured inside E1
                    pairs.append([e1start, e1end])
                elif e1start >= e2start and e1end <= e2end:
                    # E1 occured inside e2
                    pairs.append([e2start, e2end])
                elif e1start <= e2start and e2start <= e1end <= e2end:
                    # E1 preceeded and overlapped e2
                    pairs.append([e1start, e2end])
                elif e2start <= e1start and e1start <= e2end <= e1end:
                    # E2 preceeded and overlapped e1
                    pairs.append([e2start, e1end])

        # Remove duplicates from list
        uniques = []
        for p in pairs:
            if p not in uniques:
                uniques.append(p)

        # TODO: Refactor this next bit and use pandas more intelligently
        starts = [i[0] for i in uniques]
        ends = [i[1] for i in uniques]
        new_epoch = pd.DataFrame({'start_index': starts,
                                  'end_index': ends},
                                 columns=['start_index',
                                          'end_index',
                                          'epoch_name'])

        return new_epoch

    def select_epochs(self, epoch_name):
        '''
        Returns a new signal, the same as this, with everything NaN'd
        unless it is tagged with epoch_name. If epoch_name is a string,
        the self.epochs dataframe is used. If epoch_name is a dataframe,
        then it will be used instead of self.epochs.

        TODO: Examples
        '''

        if type(epoch_name) is pd.DataFrame:
            mask = epoch_name
        else:
            mask = self.epochs['epoch_name'] == epoch_name

        matched_epochs = self.epochs[mask]
        samples = matched_epochs['end_index'] - matched_epochs['start_index']

        old_data = self.as_continuous()
        new_data = np.full(old_data.shape, np.nan)
        for s in samples:
            start = s['start_index']
            end = s['end_index']
            new_data[start:end] = old_data[start:end]

        return self._modified_copy(new_data)

    def match_epochs(self, epoch_name_regex):
        '''
        Return a list of all epochs matching epoch_name_regex
        '''
        mask = self.epochs['epoch_name'].str.match(epoch_name_regex)
        df = self.epochs[mask]
        unique_epoch_names = df['epoch_name'].unique()
        return unique_epoch_names

    def multi_extract_epochs(self, list_of_epoch_names):
        '''
        Returns a dict mapping epochs from list_of_epoch_names
        to the 3D matrices created by .extract_epochs(). This function is
        particularly useful when used with its inverse, .replace_epochs().
        '''
        d = {ep: self.extract_epochs(ep) for ep in list_of_epoch_names}
        return d

    def replace_epochs(self, replacement_dict):
        '''
        Returns a new signal, created by replacing every occurrence of epochs
        in this signal with whatever is found in the replacement_dict under
        the same epoch_name.

        If the replacement matrix shape is not the same as the original
        epoch being replaced, an exception will be thrown.

        If overlapping epochs are defined, then they will be replaced in
        the order present in the epochs dataframe (i.e. sorting your
        epochs dataframe may change the results you get!). But it is a bad
        idea to replace overlapping epochs in a single operation anyway
        '''
        if self.epochs is None:
            m = "Signal.epochs must be defined in order to replace epochs"
            raise ValueError(m)

        if not len(replacement_dict):
            m = "replacement_dict must be defined in order to replace epochs"
            raise ValueError(m)

        rows = self.epochs['epoch_name'].isin(replacement_dict.keys())
        epochs_to_replace = self.epochs[rows]

        if not len(epochs_to_replace):
            m = 'No matching epochs found. Unable to replace.'
            raise ValueError(m)

        mat = self.as_continuous()

        # Define a little lambda to work on mat
        def replacer(row):
            newmat = replacement_dict[row['epoch_name']]
            mat[:, row['start_index']:row['end_index']] = newmat

        # Now, mutate mat!
        epochs_to_replace.apply(replacer, axis=1)

        return self._modified_copy(mat)

    @property
    def shape(self):
        return self._matrix.shape


# def sanity_check_epochs(self, epoch_name):
#     '''
#     There are several kinds of pathological epochs:
#       1. Epochs with NaN for a start or end time
#       2. Epochs where start comes after the end
#       3. Epochs which are completely identical triplets
#     This function searches for those and throws exceptions about them.
#     '''
#     # TODO
#     pass

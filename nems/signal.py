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
                 ('start_index', 'end_index', 'epoch_name')
            denoting the start and end of the time of an epoch, and what
            it is named. You may reuse the same name (because several epochs
            might correspond to the same stimulus, for example). start_index
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

        np.savetxt(csvfilepath, self.as_continuous(), delimiter=",", fmt=fmt)
        # TODO:
#        if isinstance(self.epochs, pd.DataFrame):
#            np.savetxt(epochfilepath, self.epochs, delimiter=",")

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
        # TODO: @Ivar test_signal_save_load was failing due to
        #       matrix shape being 200x3 instead of 3x200 on load.
        #       Saw this swapaxes line, and removing it causes the test to pass.
        #       However, removing it causes signals to not load correctly
        #       when using actual data instead of the test data.
        #       So I guess either the save method or mat_to_csv needs an
        #       axis swap somewhere as well? Wasn't sure so I figured I
        #       would leave this for you. --jacob
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

    def as_trials(self):
        '''
        Return data as a 3D array of channel x trial x time

        If trials are of uneven length, pads shorter trials with NaN. All trials
        are aligned to the start.
        '''
        if self.epochs is None:
            raise ValueError("Cannot reshape into trials without epochs info.\n"
                             "To create default trial epochs, set epochs = "
                             "signal.trial_epochs_from_reps(nreps=#). This can "
                             "not be done automatically safely in all cases. ")

        return self.fold_by('trial')

    def as_average_trial(self):
        '''
        Return data as a 2D array of channel x time averaged across trials
        '''
        m = self.as_trials()
        return np.nanmean(m, axis=0)

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

    def split_at_epoch(self, fraction):
        '''
        Returns a tuple of two signals split at fraction (rounded to the
        nearest epoch) of the original signal. If you had 10 epochs of T time
        samples, and split it at fraction=0.81, this would return (A, B) where
        A is the first eight epochs and B are the last two epochs.
        '''

        # Get time index that the fraction corresponds to
        ntimes_idx = max(1, int(self.ntimes * fraction))

        # Epochs under the time go 'left', over go 'right'
        # time index is rounded to the max index of the 'left' set
        mask = self.epochs['end_index'] <= ntimes_idx
        lepochs = self.epochs.loc[mask]
        i = lepochs['end_index'].astype('i').max()
        mask = self.epochs['start_index'] >= i
        repochs = self.epochs.loc[mask]
        # Get any epochs that were left out of left and right set.
        # i.e. their start is before left's end, and their end is after
        # right's start.
        mask = self.epochs['start_index'] < i
        mepochs = self.epochs.loc[mask]
        mask = mepochs['end_index'] > i
        mepochs = mepochs[mask]
        # Rip dataframe into two halves, one with copied starts and another
        # with copied ends, both with copied epoch names.
        # Replace missing ends and starts, respectively, with i.
        n = mepochs.shape[0]
        lsplit = mepochs.copy()
        rsplit = mepochs.copy()
        lsplit['end_index'] = [i]*n
        rsplit['start_index'] = [i]*n
        lepochs = lepochs.append(lsplit, ignore_index=True)
        repochs = repochs.append(rsplit, ignore_index=True)

        data = self.as_continuous()
        ldata = data[..., :i]
        rdata = data[..., i:]

        # Correct the index for the latter data
        repochs[['start_index', 'end_index']] -= i

        lsignal = self._modified_copy(data=ldata, epochs=lepochs)
        rsignal = self._modified_copy(data=rdata, epochs=repochs)
        return lsignal, rsignal

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

        data = self.as_continuous()
        ldata = data[..., :split_idx]
        rdata = data[..., split_idx:]

        if self.epochs is None:
            lepochs = None
            repochs = None
        else:
            mask = self.epochs['end_index'] < split_idx
            lepochs = self.epochs.loc[mask]
            mask = self.epochs['start_index'] > split_idx
            repochs = self.epochs.loc[mask]
            repochs[['start_index', 'end_index']] -= split_idx

        lsignal = self._modified_copy(ldata, epochs=lepochs)
        rsignal = self._modified_copy(rdata, epochs=repochs)

        return lsignal, rsignal

    def jackknifed_by_epochs(self, epoch_name, nsplits, split_idx, invert=False):
        '''
        Returns a new signal, with epochs matching  NaN'd out.
        Optional argument 'invert' causes everything BUT the matched epochs
        to be NaN'd. If no epochs are found that match the regex, an exception
        is thrown. The epochs data structure itself is not changed.
        '''
        mask = self.epochs['epoch_name'] == epoch_name
        matched_epochs = self.epochs[mask]

        if not matched_epochs.size:
            m = 'No epochs found matching that epoch_name. Unable to jackknife.'
            raise ValueError(m)

        m = self.as_continuous()
        if invert:
            mask = np.ones_like(m, dtype=np.bool)
        else:
            mask = np.zeros_like(m, dtype=np.bool)
        for _, row in matched_epochs.iterrows():
            lower, upper = row[['start_index', 'end_index']].astype('i')
            mask[:, lower:upper] = 0 if invert else 1

        m[mask] = np.nan
        return self._modified_copy(m)

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
    def fold_by(self, epoch_name):
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

           assert(signal.fold_by('trial').shape == (3, 3, 20))

        i.e. 3 epochs x 3 channels x 20 time samples (longest). The three
        epochs would contain 10, 15, and 0 NaN values, respectively.
        """
        if self.epochs is None:
            m = "Signal.epochs must be defined in order to fold by epochs"
            raise ValueError(m)

        mask = self.epochs['epoch_name'] == (epoch_name)
        matched_epochs = self.epochs[mask]

        if not len(matched_epochs):
            m = 'No matching epochs found. Unable to fold.'
            raise ValueError(m)

        samples = matched_epochs['end_index'] - matched_epochs['start_index']
        n_epochs = matched_epochs.shape[0]
        n_samples = samples.max().astype('i')

        folded_data = np.full((n_epochs, self.nchans, n_samples), np.nan)
        for i, (_, row) in enumerate(matched_epochs.iterrows()):
            lower, upper = row[['start_index', 'end_index']].astype('i')
            samples = upper - lower
            folded_data[i, :, :samples] = self._matrix[:, lower:upper]

        return folded_data


    def trial_epochs_from_reps(self, nreps=1):
        """
        Creates a generic epochs DataFrame with a number of trials
        based on sample length and number of repetitions specified.

        Ex: If signal._matrix has shape 3x100,
            trial_epochs_from_reps(nreps=5) would generate a DataFrame of
            of the form:
                {'start_index': [0, 20, 40, 60, 80],
                 'end_index': [20, 40, 60, 80, 100],
                 'epoch_name': ['trial0', 'trial1', 'trial2',
                                'trial3', 'trial4']}

        Note: If the number of time samples is not evenly divisible by
              the number of repetitions, then an additional trial will be
              added to carry the remainder of the time samples.
              (i.e. if there are 100 time samples and nreps=3, there will
              be 3 trials of length 33 and a 4th trial of length 1)
              TODO: is this good default behavior, or would it be better to
                    either throw an error, chop off the remainder, or
                    append the remainder to the nth trial?

        Reminder: epochs indices behave similar to python list indices, so
                  start_index is inclusive while end_index is exclusive,
                  making the actual index values 0-99.

        TODO: finish doc

        TODO: Some way to get reps info from the data instead of
              requiring user-provided?

              Could use a default value based on length, like
              nreps = np.ceil(n_samples/10) (and last trial would be
              shorter than the others if n_samples not multiple of 10).

        """

        trial_size = int(self.ntimes/nreps)
        remainder = self.ntimes % nreps

        starts = []
        ends = []
        names = []

        for i in range(nreps):
            start = (i)*trial_size
            end = (i+1)*trial_size
            starts.append(start)
            ends.append(end)
            names.append('trial')
        if remainder:
            start = (nreps)*trial_size
            end = start+remainder
            starts.append(start)
            ends.append(end)
            names.append('trial')

        epochs = pd.DataFrame({'start_index': starts,
                           'end_index': ends,
                           'epoch_name': names},
                          columns=['start_index', 'end_index', 'epoch_name'])

        return epochs

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

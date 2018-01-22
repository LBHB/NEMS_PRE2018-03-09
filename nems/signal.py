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
        if self.epochs:
            np.savetxt(epochfilepath, self.epochs, delimiter=",")

        with open(jsonfilepath, 'w') as fh:
            attributes = self._get_attributes()
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

        # TODO: should self.trial_epochs_from_nreps be baked in here instead?
        #       i.e. if no epochs, get some default ones instead of error
        if self.epochs is None:
            raise ValueError("Cannot reshape into trials without epochs info.\n"
                             "For default trial epochs, try setting epochs = "
                             "signal.trial_epochs_from_reps(nreps='#')")

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
        Returns a tuple of 2 new signals; because this may split one of the
        repetitions unevenly, it sets the nreps to 1 in both of the new signals.
        '''

        if self.epochs is None:
            # TODO: Best way to handle this? Doesn't seem like epochs
            #       should be required for split_at_time, but have issues
            #       if it isn't defined.
            raise ValueError("signal.epochs must be defined in order to split")

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

    def jackknifed_by_epochs(self, regex, invert=False):
        '''
        Returns a new signal, with entire epochs NaN'd out. Optional argument
        'invert' causes everything BUT the matched epochs to be NaN.
        '''

        mask = self.epochs['epoch_name'].str.contains(regex)
        matched_epochs = self.epochs[mask]
        # TODO: best way to report no matches?
        #if not matched_epochs.size:
        #    return np.empty((1,1))

        m = self.as_continuous()
        if invert:
            mask = np.ones_like(m, dtype=np.bool)
        else:
            mask = np.zeros_like(m, dtype=np.bool)
        for _, row in matched_epochs.iterrows():
            lower, upper = row[['start_index', 'end_index']].astype('i')
            mask[:, lower:upper] = 0 if invert else 1

        m[mask] = np.nan
        # TODO: should the epochs be removed from self.epochs as well?
        #       or add on an extra column to tag them as NaN'd?
        #       For now just leaving them - seems better to have that info
        #       (i.e. so user can see that the NaN'd sections
        #        line up with trials 3, 7, 9 or w/e)
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
    def fold_by(self, regex):
        """Returns matrix with shape epochs x channels x time, wherever
        epoch_name in signal.epochs matches the provided regular expression.
        If a plain string is provided instead of a regular expression, the
        matching behavior will be similar to python's 'in' operator.
        For epochs with uneven length, NaNs will be appended to the shorter
        lengths.

        Ex: with signal.epochs = {'start_index': [0, 10, 15],
                                  'end_index': [10, 15, 35],
                                  'epoch_name': [trial1, trial2, trial3]}
            number of channels = 3,
            fold_by('trial') would return a matrix of shape
            (3, 3, 20), for 3 epochs x 3 channels x 20 time samples (longest).
            The epochs would contain 10, 15, and 0 NaN values, respectively.

        TODO: finish doc

        """

        if self.epochs is None:
            raise ValueError("Signal.epochs must be defined in order"
                             "to fold by epochs")
        mask = self.epochs['epoch_name'].str.contains(regex)
        matched_epochs = self.epochs[mask]
        # TODO: best way to report no matches?
        #if not len(matched_epochs):
        #    return np.empty((1,1))
        samples = matched_epochs['end_index'] - matched_epochs['start_index']
        n_epochs = matched_epochs.shape[0]
        n_samples = samples.max().astype('i')

        folded_data = np.full((n_epochs, self.nchans, n_samples), np.nan)
        for i, (_, row) in enumerate(matched_epochs.iterrows()):
            lower, upper = row[['start_index', 'end_index']].astype('i')
            samples = upper - lower
            folded_data[i, :, :samples] = self._matrix[:, lower:upper]

        return folded_data

#       TODO:
#       1) specify list of regexp instead of one string,
#          along with 'logic' and 'action' specs.
#          ex: regex=['^stim', '^trial', '^rep'], logic='OR',
#              action=None
#
#              would match all epochs with any of the above
#              patterns as separate folds/trials/whatever.
#              (useful if naming scheme not known or different
#               schemes used interchangeably)
#
#          ex: regex=['trial2', '^pupil_closed'], logic='AND',
#              action=my_function_object
#
#              not sure exactly how this would work yet, but the idea
#              is that you could specify an action to take on the final
#              data returned, like changing the values of the samples
#              in 'trial2' to nan where it lines up with 'pupil_closed'
#              then drop the latter epoch from the folding.
#              I guess a copy of _matrix would have to be passed
#              to the callback? Maybe not feasible.
#
#        why not just put the OR / AND in the regex?
#           -not as intuitive for people that aren't familar with regexp
#           -also complicates the fold_by code because
#                matching more than one epoch to the same time period
#                would necessitate copying parts of the data
#                (or doing something else, but simple stacking
#                 won't work).
#
#       possible solutions:
#           1) make user provide a function that decides what to
#              do with the matches (would necessitate knowledge
#              of data structure details, so not ideal).
#           2) make order of regexs list matter.
#              ex: ['trial', 'pupil_closed', 'stim1'], 'AND'
#                  would fold by trial, but only for trials that
#                  overlap with the pupil_closed and stim1 epochs.
#                  ['stim', 'pupil_closed', 'trial5'], 'AND'
#                  would fold by stim, but only for stims that
#                  overlap with the pupil_closed and trial5 epochs.
#           3) but what if want to fold by just the overlapping slice?
#              separate functions? fold_by_or vs fold_by_or_filter?
#              ex: ['trial', 'pupil_closed'], 'AND'
#                  would match all trials, but slice out only the
#                  portions that overlap with pupil_closed.
#
#        started fold_by_or and fold_by_and below


    def fold_by_or(self, regexs):
        # TODO: make a single regex string that
        #       matches to one or more of the patterns
        combined_regex = 'regexs0 | regexs1 | etc...'
        return self.fold_by(combined_regex)

    def fold_by_and(self, regexs):
        # TODO
        combined_regex = 'regexs0 && regexs1 && etc...'
        return self.fold_by(combined_regex)

    def trial_epochs_from_reps(self, nreps=1):
        """Creates a generic epochs DataFrame with a number of trials
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
            names.append('trial%d'%i)
        if remainder:
            start = (nreps)*trial_size
            end = start+remainder
            starts.append(start)
            ends.append(end)
            names.append('trial%d'%nreps)

        epochs = pd.DataFrame({'start_index': starts,
                           'end_index': ends,
                           'epoch_name': names},
                          columns=['start_index', 'end_index', 'epoch_name'])

        return epochs



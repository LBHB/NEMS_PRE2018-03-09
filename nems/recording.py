import os
import logging
from .signal import Signal


class Recording:

    def __init__(self, signals):
        '''
        Signals argument should be a dictionary of signal objects.
        '''
        self.signals = signals

        # Verify that all signals are from the same recording
        recordings = [s.recording for s in self.signals.values()]
        if not len(set(recordings)) == 1:
            raise ValueError('Not all signals are from the same recording!')
        self.name = recordings[0]

    # Defining __getitem__ and __setitem__ make recording objects behave
    # like dictionaries when subscripted. e.g. recording['signal_name']
    # instead of recording.get_signal('signal_name').
    # See: https://docs.python.org/3/reference/datamodel.html?emulating-container-types#emulating-container-types

    def __getitem__(self, key):
        return self.get_signal(key)

    def __setitem__(self, key, val):
        val.name = key
        self.add_signal(val)

    @staticmethod
    def load(directory):
        '''
        Loads all the signals (CSV/JSON pairs) found in DIRECTORY and
        returns a Recording object containing all of them.
        '''
        files = Signal.list_signals(directory)
        basepaths = [os.path.join(directory, f) for f in files]
        signals = [Signal.load(f) for f in basepaths]
        signals_dict = {s.name: s for s in signals}
        return Recording(signals=signals_dict)

    def save(self, directory, no_subdir=False):
        '''
        Saves all the signals (CSV/JSON pairs) in this recording into
        DIRECTORY in a new directory named the same as this recording.
        If optional argument no_subdir=True is provided, it
        will not create the subdir.
        '''
        if not no_subdir:
            directory = os.path.join(directory, self.name)
        print(directory)
        try:
            os.mkdir(directory)
        except:
            pass
        for s in self.signals.values():
            s.save(directory)
        pass

    def get_signal(self, signal_name):
        '''
        Returns the signal object with the given signal_name, or None
        if it was was found.

        signal_name should be a string
        '''
        if signal_name in self.signals:
            return self.signals[signal_name]
        else:
            return None

    def add_signal(self, signal):
        '''
        Adds the signal equal to this recording. Any existing signal
        with the same name will be overwritten. No return value.
        '''
        if not isinstance(signal, Signal):
            raise TypeError("Recording signals must be instances of"
                            "of class Signal.")
        self.signals[signal.name] = signal

    def split_at_time(self, fraction):
        '''
        Calls .split_at_time() on all signal objects in this recording.
        For example, fraction = 0.8 will result in two recordings,
        with 80% of the data in the left, and 20% of the data in
        the right signal. Useful for making est/val data splits, or
        truncating the beginning or end of a data set.
        '''
        left = {}
        right = {}
        for s in self.signals.values():
            (l, r) = s.split_at_time(fraction)
            left[l.name] = l
            right[r.name] = r
        return (Recording(signals=left), Recording(signals=right))

    def jackknife_by_epoch(self, regex, signal_names=None,
                           invert=False):
        '''
        By default, calls jackknifed_by_epochs on all signals and returns a new
        set of data. If you would only like to jackknife certain signals,
        while copying all other signals intact, provide their names in a
        list to optional argument 'only_signals'.
        '''
        raise NotImplementedError
        # if signal_names is not None:
        #     signals = {n: self.signals[n] for n in signal_names}
        # else:
        #     signals = self.signals

        # kw = dict(regex=regex, invert=invert)
        # split = {n: s.jackknifed_by_epochs(**kw) for n, s in signals.items()}
        # return Recording(signals=split)

    def jackknife_by_time(self, nsplits, split_idx, only_signals=None,
                          invert=False):
        '''
        By default, calls jackknifed_by_time on all signals and returns a new
        set of data.  If you would only like to jackknife certain signals,
        while copying all other signals intact, provide their names in a
        list to optional argument 'only_signals'.
        '''
        raise NotImplementedError        # TODO
        # new_sigs = {}
        # for sn in self.signals.keys():
        #     if (not only_signals or sn in set(only_signals)):
        #         s = sn
        #         new_sigs[sn] = s.jackknifed_by_time(nsplits, split_idx,
        #                                             invert=invert)
        # return Recording(signals=new_sigs)

    def jackknifes_by_epoch(self, nsplits, epoch_name, only_signals=None):
        raise NotImplementedError         # TODO

    def jackknifes_by_time(self, nsplits, only_signals=None):
        raise NotImplementedError         # TODO

    def concatenate_recordings(self, recordings):
        '''
        Concatenate more recordings on to the end of this Recording,
        and return the result. Recordings must have identical signal
        names, channels, and fs, or an exception will be thrown.
        '''
        signal_names = self.signals.keys()
        for recording in recordings:
            if signal_names != recording.signals.keys():
                raise ValueError('Recordings do not contain same signals')

        # Merge the signals and return it as a new recording.
        merged_signals = {}
        for signal_name in signal_names:
            signals = [r.signals[signal_name] for r in recordings]
            merged_signals[signal_name] = Signal.concatenate_time(signals)

        # TODO: copy the epochs as well
        raise NotImplementedError    # TODO

        return Recording(merged_signals)

        # TODO: copy the epochs as well
    def select_epoch():
        raise NotImplementedError    # TODO

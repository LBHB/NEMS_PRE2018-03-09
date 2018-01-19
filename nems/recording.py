import os
from .signal import Signal


class Recording:

    def __init__(self, signals):
        self.signals = signals

        # Verify that all signals are from the same recording
        recordings = [s.recording for s in self.signals.values()]
        if not len(set(recordings)) == 1:
            raise ValueError('Not all signals are from the same recording!')
        self.name = recordings[0]

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
        os.mkdir(directory)
        for s in self.signals.values():
            s.save(directory)
        pass

    def split_at_rep(self, fraction):
        '''
        Calls split_at_rep() on all signal objects in this recording.
        '''

        left = {}
        right = {}
        for s in self.signals.values():
            (l, r) = s.split_at_rep(fraction)
            left[l.name] = l
            right[r.name] = r
        return (Recording(signals=left), Recording(signals=right))

    def split_at_time(self, fraction):
        '''
        Calls split_at_time() on all signal objects in this recording.
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

    def jackknifed_by_reps(self, nsplits, split_idx, signal_names=None,
                           invert=False):
        '''
        By default, calls jackknifed_by_reps on all signals and returns a new
        set of data. If you would only like to jackknife certain signals,
        provide their names in a list to optional argument 'only_signals'.
        '''
        if signal_names is not None:
            signals = {n: self.signals[n] for n in signal_names}
        else:
            signals = self.signals

        kw = dict(nsplits=nsplits, split_idx=split_idx, invert=invert)
        split = {n: s.jackknifed_by_reps(**kw) for n, s in signals.items()}
        return Recording(signals=split)

    def jackknifed_by_time(self, nsplits, split_idx, only_signals=None,
                           invert=False):
        '''
        By default, calls jackknifed_by_time on all signals and returns a new
        set of data. If you would only like to jackknife certain signals,
        provide their names in a list to optional argument 'only_signals'.
        '''

        new_sigs = {}
        for sn in self.signals.keys():
            if (not only_signals or sn in set(only_signals)):
                s = sn
                new_sigs[sn] = s.jackknifed_by_time(nsplits, split_idx,
                                                    invert=invert)
        return Recording(signals=new_sigs)


    # def get_interval(self, interval):
    #     '''
    #     Given an interval tuple ("name", "start", "stop"), returns a new
    #     recording object of just the data at that point.
    #     '''
    #     pass

    @classmethod
    def concatenate_recordings(cls, recordings):
        # Make sure they all contain the same set of signals. If not, this is
        # undefined behavior.
        signal_names = recordings[0].signals.keys()
        for recording in recordings:
            if signal_names != recording.signals.keys():
                raise ValueError('Recordings do not contain same signals')

        # Merge the signals and return it as a new recording.
        merged_signals = {}
        for signal_name in signal_names:
            signals = [r.signals[signal_name] for r in recordings]
            merged_signals[signal_name] = Signal.concatenate_time(signals)

        return Recording(merged_signals)

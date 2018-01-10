import signal


class Recording():
    
    def __init__(self, **kwargs):
        '''
        A Recording is a collection of signals that were recorded simultaneously,
        such that their time indexes line up (and thus the time dimension of all
        signals should be the same length and have the same sampling rate).

        TODO:

        Recordings have several useful methods:
        .split_at_time()
        .just_interval()

        
        '''
        self.signals = kwargs['signals']

    @staticmethod
    def load(directory):
        '''
        Loads all the signals (CSV/JSON pairs) found in DIRECTORY and
        returns a Recording object containing all of them.
        '''
        files = list_signals_in_dir(directory)
        filepaths = [os.path.join(directory, f) for f in files]
        signals = [Signal.load(f) for f in basepaths]
        signals_dict = {s.name: s for s in signals}
        return Recording(signals=signals_dict)

    def get_interval(self, interval):
        '''
        Given an interval tuple ("name", "start", "stop"), returns a new
        recording object of just the data at that point.
        '''
        pass
        

    def split_at_time(self, fraction):
        '''
        Splits a Recording into two Recordings, with each signal being split
        at the same point in the the time series. For example, fraction = 0.8
        splits 80% of the data into the left, and 20% of the data 
        into the right signal. Useful for making est/val data splits, or 
        truncating the beginning or end of a data set. 
        '''
        left = {}
        right = {}
        for s in self.signals.values():
            (l, r) = s.split_by_time(fraction)
            left[l.name] = l
            right[r.name] = r
        return (Recording(signals=left), Recording(signals=right))



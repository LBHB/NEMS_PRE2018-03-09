import os
import json
import pandas as pd   # Pulled in for fast CSV i/o
import numpy as np


# NOTE! THIS CODE IN HORRIBLE, UNTRUSTWORTHY FLUX UNTIL: Tues, Jan 9th

class Signal():
    '''
    A signal has these required fields:
    .name         The name of the signal, e.g. 'stim' or 'resp-a1' [string]
    .recording    The name of the recording session [string]
    .fs           The frequency [uint] of sampling, in Hz.

    There is one optional initialization field:
    .meta         Metadata for the signal

    Internally, the signal has several matrix dimensions you may refer to:
    .nchans       The number of input channels [uint]
    .ntimes       The number of time samples per trial [uint]

    TODO: CHANGE THIS INTERNALLY
    .nreps        The number of trial repetitions [uint]

    To get (cheap) views of the same data matrix in different ways, use:
    .as_single_trial()
    .as_average_trial()

    To create a mutated copies of this object (a more expensive op), use:
    .split_by_reps(fraction)
    .split_by_time(fraction)
    .jackknifed_by_reps(nsplits, split_idx)
    .jackknifed_by_time(nsplits, split_idx)
    .where(element_condition_function1, condition2, ...)
    .normalized()

    To combine this Signal with other Signals, use:
    .append_timeseries()
    .append_reps()
    .append_channels()

    The data is hidden inside:
    .__matrix__   A matrix of time x channels [np.ndarray]

    TODO:
    - [ ] Make the LAST dimension time in the signal object matrix?
    - [ ] Remove 'recording', cellid, nreps from signals?
    - [ ] Append timeseries
    - [ ] Normalize()
    '''

    def __init__(self, signal_name, recording, fs, matrix, nreps=None, meta=None):
        # Four required parameters:
        self.name = signal_name
        self.recording = recording
        self.fs = fs
        self.nreps = nreps
        self.__matrix__ = matrix

        self.meta = meta

        # TODO: More error checking on all of these:
        if type(self.name) is not str:
            raise ValueError('Name of signal must be a string:'
                             + str(self.name))

        if type(self.recording) is not str:
            raise ValueError('Recording must be a string:'
                             + str(self.recording))

        if type(self.fs) is not int or self.fs < 1:
            raise ValueError('fs of signal must be a positive integer:'
                             + str(self.fs))

        if type(self.__matrix__) is not np.ndarray:
            raise ValueError('matrix must be a np.ndarray:'
                             + type(self.__matrix__))

        # self.__matrix__.flags.writeable = False  # Make it immutable

        # TODO: Rearrange this matrix dimensions; I'm not sure T,C,R is ideal
        n_channels, n_time = self.__matrix__.shape

        if n_time < n_channels:
            raise ValueError(('Matrix dims weird: (T, C) = '
                              + str((T, C))
                              + '; failing because we expected a long time'
                              + ' series but found T < C'))

        self.nchans = n_channels
        self.ntimes = n_time

    def savetocsv(self, dirpath, fmt=None):
        """ Saves this signal to a CSV as a single long trial. If
        desired, you may use optional parameter fmt='%1.3f' to alter
        the precision of the matrices written to file"""
        filename = self.recording + '_' + self.name
        filepath = os.path.join(dirpath, filename)
        csvfilepath = filepath + '.csv'
        jsonfilepath = filepath + '.json'
        if fmt:
            np.savetxt(csvfilepath,
                       self.as_single_trial(),
                       delimiter=", ",
                       fmt=fmt)
        else:
            np.savetxt(csvfilepath,
                       self.as_single_trial(),
                       delimiter=", ")
        # df = pd.DataFrame(self.as_single_trial())  # 10x slower than savetxt
        obj = {'name': self.name,
               'recording': self.recording,
               'fs': self.fs,
               'nchans': self.nchans,
               'nreps': self.nreps,
               'ntimes': self.ntimes,
               'meta': self.meta}
        with open(jsonfilepath, 'w') as f:
            json.dump(obj, f)
        return (csvfilepath, jsonfilepath)

    def copy(self):
        """ Returns a copy of the matrix; TODO: Tests """
        return self.__matrix__.copy()

    def modified_copy(self, m):
        """ Returns a copy of the Signal using the modified data matrix m"""
        new_obj = Signal(signal_name=self.name,
                         recording=self.recording,
                         fs=self.fs,
                         meta=self.meta,
                         matrix=m)
        return new_obj

    def single_to_multi_trial(self, m):
        """ TODO """
        mat = m.reshape(self.nreps, self.ntimes, self.nchans)
        mat = mat.swapaxes(1, 0)
        mat = mat.swapaxes(2, 1)
        return mat

    def get_matrix(self):
        """ Just return self.__matrix__, after doing any other checks needed.
        TODO: anything else needed here?
        """
        return self.__matrix__

    def as_single_trial(self):
        """ Return the data by concatenating all reps one after another
        so that it appears to be a single, long trial. (i.e. 1 repetition)  """
        mat = self.__matrix__
        #mat = self.__matrix__.swapaxes(1, 0)
        #mat = mat.swapaxes(2, 0)
        #mat = mat.reshape(self.ntimes * self.nreps, self.nchans)
        return mat

    def as_average_trial(self):
        """ Return the matrix as the average of all repetitions.  """
        return np.nanmean(self.__matrix__, axis=2)

    def as_old_matlab_format(self):
        """ Return the matrix as the average of all repetitions.  """
        m = self.__matrix__
        m = m.swapaxes(1, 2)
        m = m.swapaxes(0, 2)
        return m

    def split_by_reps(self, fraction):
        """ Returns a tuple of 2 new signals; TODO"""
        # TODO
        pass

    def split_by_time(self, fraction):
        """ Returns a tuple of 2 new signals; TODO """
        m = self.as_single_trial()
        split_idx = max(1, int(self.ntimes * fraction))
        left = m[0:split_idx, :]
        right = m[split_idx:, :]
        l = Signal(signal_name=self.name,
                   recording=self.recording,
                   nreps=1,
                   fs=self.fs,
                   meta=self.meta,
                   matrix=left)
        r = Signal(signal_name=self.name,
                   recording=self.recording,
                   nreps=1,
                   fs=self.fs,
                   meta=self.meta,
                   matrix=right)
        return (l, r)

    def jackknifed_by_reps(self, nsplits, split_idx):
        """ Returns a new signal, with entire reps NaN'd out. If nreps is not
        an integer multiple of nsplits, an error is thrown. """
        ratio = (self.nreps / nsplits)
        if ratio != int(ratio):
            raise ValueError('nreps must be an integer multiple of nsplits:'
                             + str(ratio))
        m = self.__matrix__.copy()
        m[:, :, split_idx] = float('NaN')
        return self.modified_copy(m)

    def jackknifed_by_time(self, nsplits, split_idx):
        """ Returns a new signal, with some data NaN'd out based on its position
        in the time stream. split_idx is indexed from 0; if you have 20 splits,
        the first is #0 and the last is #19. """
        n_time_elements = self.ntimes * self.nreps
        splitsize = int(n_time_elements / nsplits)
        if splitsize < 1:
            raise ValueError('Too many jackknifes? Splitsize was: '
                             + str(splitsize))
        split_start = split_idx * splitsize
        if split_idx == nsplits - 1:
            split_end = n_time_elements
        else:
            split_end = (split_idx + 1) * splitsize
        m = self.as_single_trial().copy()
        m[split_start:split_end, :] = float('NaN')
        return self.modified_copy(self.single_to_multi_trial(m))

    def append_timeseries(self, other):
        """ TODO """
        if not type(other) == type(self):
            raise ValueError('append_timeseries needs another Signal object.')
        pass

    def append_reps(self, other):
        """ TODO """
        if not type(other) == type(self):
            raise ValueError('append_reps needs another Signal object.')
        pass

    def append_channels(self, other):
        """ TODO """
        if not type(other) == type(self):
            raise ValueError('append_reps needs another Signal object.')

    def where(self, condition):
        """Returns a new signal, with data that does not meet the condition
        NaN'd out."""
        # TODO
        pass

    def normalized(self):
        """ Returns a new signal, same as this one, but shifted to
        have zero mean and unity variance on each channel."""
        # TODO
        # p = self.as_single_trial()
        # chan_means =
        # chan_variances =
        # m = self.__matrix__
        # obj = Signal(matrix=m)
        # return obj
        pass


def load_from_file(basepath):
    """ Loads the CSV & JSON files and returns a Signal() object.
    Example: If you want to load
      /tmp/sigs/gus027b13_p_PPS_resp-a1.csv
      /tmp/sigs/gus027b13_p_PPS_resp-a1.json
    then give this function
      /tmp/sigs/gus027b13_p_PPS_resp-a1"""
    csvfilepath = basepath + '.csv'
    jsonfilepath = basepath + '.json'
    # Weirdly, numpy is 10x slower than read_csv (pandas):
    # mat = np.loadtxt(csvfilepath, delimiter=", ")
    mat = pd.read_csv(csvfilepath, header=None).values
    mat = mat.astype('float')
    with open(jsonfilepath, 'r') as f:
        js = json.load(f)
        # NOTE: Please also see 'single_to_multi_trial';
        # mat = mat.reshape(js['nreps'], js['ntimes'], js['nchans'])
        s = Signal(signal_name=js['name'],
                   recording=js['recording'],
                   fs=js['fs'],
                   nreps=js['nreps'],
                   meta=js['meta'],
                   matrix=mat)
    return s


def load_signals(basepaths):
    """ Returns a list of the Signal objects created by loading the
    signal files at paths. """
    signals = [load_from_file(f) for f in basepaths]
    return signals


def list_signals_in_dir(dirpath):
    """ Returns a list of all CSV/JSON signal files found in dirpath.
    Returns relative paths; not absolute ones. """
    files = os.listdir(dirpath)
    just_fileroot = lambda f: os.path.splitext(os.path.basename(f))[0]
    csvs = [just_fileroot(f) for f in files if f.endswith('.csv')]
    jsons = [just_fileroot(f) for f in files if f.endswith('.json')]
    overlap = set.intersection(set(csvs), set(jsons))
    return overlap


def load_signals_in_dir(dirpath):
    """ Returns a dict of all CSV/JSON signals found in BASEPATH. """
    files = list_signals_in_dir(dirpath)
    filepaths = [os.path.join(dirpath, f) for f in files]
    signals = load_signals(filepaths)
    signals_dict = {s.name: s for s in signals}
    return signals_dict


def split_signals_by_time(signals, fraction):
    """ Splits a dict of signals into two dicts of signals, with
    each signal split at the same point in the the time series. For example,
    fraction = 0.8 splits 80% of the data into the left, and 20% of the data
    into the right signal. Useful for making est/val data splits, or truncating
    the beginning or end of a data set. Note: Trial information is lost!"""
    left = {}
    right = {}
    for s in signals.values():
        (l, r) = s.split_by_time(fraction)
        left[l.name] = l
        right[r.name] = r
    return (left, right)


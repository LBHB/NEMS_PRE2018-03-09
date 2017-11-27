import os
import json
import numpy as np


class Signal():
    """ A signal is immutable, and has these required fields:
    .name         The name of the signal, e.g. 'stim' or 'a1-resp' [string]
    .recording    The name of the recording session [string]
    .cellid       The name of the cellid [string]
    .fs           The frequency [uint] of sampling, in Hz.
    .nchans       The number of input channels
    .nreps        The number of trial repetitions
    .ntimes       The number of time samples per trial

    There is one optional field:
    .meta         Metadata for the signal

    To get (cheap) views of the same data matrix in different ways, use:
    .as_single_trial()
    .as_average_trial()

    To create (more expensive) modified copies of this object, use:
    .jackknifed_by_reps(nsplits, split_idx)
    .jackknifed_by_time(nsplits, split_idx)
    .with_condition(condition)
    .normalized()

    To combine with other objects, use:
    .append_timeseries()
    .append_reps()
    .append_channels()

    The immutable data is hidden inside:
    .__matrix__   A matrix of channels x time x repetitions [np.ndarray]
    """
    def __init__(self, **kwargs):
        # Four required parameters:
        self.name = kwargs['signal_name']
        self.cellid = kwargs['cellid']
        self.recording = kwargs['recording']
        self.fs = int(kwargs['fs'])
        self.__matrix__ = kwargs['matrix']

        # One optional parameter:
        self.meta = kwargs['meta'] if 'meta' in kwargs else None

        # TODO: More error checking on all of these:
        if type(self.name) is not str:
            raise ValueError('Name of signal must be a string:'
                             + str(self.name))

        if type(self.cellid) is not str:
            raise ValueError('Cellid must be a string:'
                             + str(self.cellid))

        if type(self.recording) is not str:
            raise ValueError('Recording must be a string:'
                             + str(self.recording))

        if type(self.fs) is not int or self.fs < 1:
            raise ValueError('fs of signal must be a positive integer:'
                             + str(self.fs))

        if type(self.__matrix__) is not np.ndarray:
            raise ValueError('matrix must be a np.ndarray:'
                             + type(self.__matrix__))

        self.__matrix__.flags.writeable = False  # Make it immutable

        (C, T, R) = self.__matrix__.shape

        if T < R or T < C:
            raise ValueError(('Matrix dims weird: (C, T, R) = '
                              + str((C, T, R))
                              + '; failing because we expected a long time'
                              + ' series but found T < R or T < C'))

        self.nchans = C
        self.ntimes = T
        self.nreps = R


    def savetocsv(self, dirpath):
        """ Saves this signal to a CSV as a single long trial. """
        filename = self.recording + '_' + self.name
        filepath = os.path.join(dirpath, filename)
        csvfilepath = filepath + '.csv'
        jsonfilepath = filepath + '.json'
        np.savetxt(csvfilepath,
                   self.as_single_trial(),
                   delimiter=", ")
        obj = {'name': self.name,
               'recording': self.recording,
               'cellid': self.cellid,
               'fs': self.fs,
               'nchans': self.nchans,
               'nreps': self.nreps,
               'ntimes': self.ntimes,
               'meta': self.meta}
        with open(jsonfilepath, 'w') as f:
            json.dump(obj, f)
        return (csvfilepath, jsonfilepath)

    def copy(self):
        """ Returns a copy of the Chans x Time x Reps matrix; TODO: Tests """
        return self.__matrix__.copy()

    def as_single_trial(self):
        """ Return the data by concatenating all reps one after another
        so that it appears to be a single, long trial. (i.e. 1 repetition)  """
        (C, T, R) = self.__matrix__.shape
        return self.__matrix__.reshape(C, T*R)

    def as_average_trial(self):
        """ Return the matrix as the average of all repetitions.  """
        return np.nanmean(self.__matrix__, axis=2)

    def jackknifed_by_reps(self, nsplits, split_idx):
        """ Returns a new signal, with entire reps NaN'd out. """
        # TODO
        pass

    def jackknifed_by_time(self, nsplits, split_idx):
        """ Returns a new signal, with some data in every trial NaN'd out."""
        # TODO
        pass

    def append_timeseries(self, other):
        """ TODO """

    def append_reps(self, other):
        """ TODO """

    def append_channels(self, other):
        """ TODO """

    def with_condition(self, condition):
        """Returns a new signal, with data that does not meet the condition
        NaN'd out."""
        # TODO
        pass

    def normalized(self):
        """ Returns a new signal, same as this one, but shifted to
        have zero mean and unity variance."""
        # TODO
        pass


def loadfromcsv(csvfilepath, jsonfilepath):
    """ Loads the CSV file. """
    mat = np.loadtxt(csvfilepath, delimiter=", ")
    with open(jsonfilepath, 'r') as f:
        js = json.load(f)
    matrix = mat.reshape(js['nchans'], js['ntimes'], js['nreps'])
    s = Signal(signal_name=js['name'],
               cellid=js['cellid'],
               recording=js['recording'],
               fs=js['fs'],
               meta=js['meta'],
               matrix=matrix)
    return s

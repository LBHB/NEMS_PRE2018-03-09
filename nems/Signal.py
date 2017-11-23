import os
import numpy as np


class Signal():
    """ A signal is immutable, and has these required fields:
    .name         The name of the signal, e.g. 'stim' or 'a1-resp' [string]
    .recording    The name of the recording session [string]
    .cellid       The name of the cellid [string]
    .fs           The frequency [uint] of sampling, in Hz.

    There is one optional field:
    .meta         Metadata for the signal

    To see the dimensions of the thing, use
    .chans_samps_trials()

    To get views of the same data in different ways, use:
    .as_single_trial()
    .as_average_trial()

    To create modified copies of this object, use:
    .jackknifed_by_trials(nsplits, split_idx)
    .jackknifed_by_time(nsplits, split_idx)
    .with_condition(condition)
    .normalized()

    To combine with other objects, use:
    .append_timeseries()
    .append_trials()
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

        (C, T, R) = self.chans_samps_trials()

        if T < R or T < C:
            raise ValueError(('Matrix dims weird: (C, T, R) = ' +
                              str((C, T, R))))

    def chans_samps_trials(self):
        """ Returns (C, T, R), where:
        C  is the number of channels
        T  is the number of time samples per trial repetition
        R  is the number of trial repetitions """
        return self.__matrix__.shape

    def savetocsv(self, dirpath):
        """ Saves this signal to a CSV. """
        np.savetxt(os.path.join(dirpath, "blah.csv"),
                   self.__matrix__,
                   delimiter=", ")

    def copy(self):
        """ Returns a copy of the Chans x Time x Reps matrix; TODO: Tests """
        return self.__matrix__.copy()

    def as_single_trial(self):
        """ Return the data by concatenating all trials one after another
        so that it appears to be a single, long trial. (i.e. 1 repetition)  """
        (C, T, R) = self.__matrix__.shape
        return self.__matrix__.reshape(C, T*R)

    def as_average_trial(self):
        """ Return the matrix as the average of all repetitions.  """
        return np.nanmean(self.__matrix__, axis=2)

    def jackknifed_by_trials(self, nsplits, split_idx):
        """ Returns a new signal, with entire trials NaN'd out. """
        # TODO
        pass

    def jackknifed_by_time(self, nsplits, split_idx):
        """ Returns a new signal, with some data in every trial NaN'd out."""
        # TODO
        pass

    def append_timeseries(self, other):
        """ TODO """

    def append_trials(self, other):
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

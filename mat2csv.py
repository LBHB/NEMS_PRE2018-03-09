import os
import numpy as np
import scipy.io

DEFAULT_DIRPATH = '/home/ivar/mat/'


class Signal():
    """ A signal has these properties:
    .name         The name of the signal [string]
    .fs           The frequency [uint] of sampling, in Hz.

    And some hidden data:
    .__matrix__   A matrix of channels x time x repetitions [numpy ndarray]
    """
    def __init__(self, **kwargs):
        self.name = kwargs['name']
        self.__matrix__ = kwargs['matrix']  # must be nparray
        self.fs = int(kwargs['fs'])

        if type(self.name) is not str:
            raise ValueError('Name of signal must be a string:'
                             + str(self.name))

        if type(self.fs) is not int or self.fs < 1:
            raise ValueError('fs of signal must be a positive integer:'
                             + str(self.fs))

        if type(self.__matrix__) is not np.ndarray:
            raise ValueError('matrix must be a np.ndarray:'
                             + type(self.__matrix__))

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

    def as_ctr_matrix(self):
        return self.__matrix__

    def as_time_stream(self):
        """ Return the matrix as a single stream of data. (1 repetition)  """
        (C, T, R) = self.__matrix__.shape
        return self.__matrix__.reshape(C, T*R)

    def as_average_trial(self):
        """ Return the matrix as the average of all repetitions.  """
        # data['avgresp']=np.transpose(data['avgresp'],(1,0))
        avg = np.nanmean(self.__matrix__, axis=2)
        return avg

    def savetocsv(self, dirpath):
        """ Saves this signal to a CSV. """
        np.savetxt(os.path.join(dirpath, "blah.csv"),
                   self.__matrix__,
                   delimiter=", ")


class MatlabRecording():
    """ A MatlabRecording is a collection of simultaneous signals.
    .name      Recording session name, usually something like 13.
    .signals   A list of all the signals
    """
    def __init__(self, **kwargs):
        m = kwargs['matdata']
        self.name = 'TODO'
        self.signals = []

        # Verify that this is not a 'stimid' recording file format
        signal_names = set(m.dtype.names)
        if 'stimids' in signal_names:
            raise ValueError('stimids are not supported yet, sorry')

        self.meta = extract_metadata(m)

        # Extract the two required signals, and possibly the pupil as well
        self.add_signal(Signal(name='stim',
                               matrix=m['stim'],
                               fs=self.meta['stimfs']))
        self.add_signal(Signal(name='resp',
                               matrix=np.swapaxes(m['resp_raster'], 0, 1),
                               fs=self.meta['respfs']))
        if 'pupil' in signal_names:
            self.add_signal(Signal(name='pupil',
                                   matrix=(np.swapaxes(m['pupil']*0.01,
                                           0, 1)),
                                   fs=self.meta['respfs']))
        # TODO: instead of respfs, switch to pupilfs and behavior_conditionfs
        if 'behavior_condition' in signal_names:
            self.add_signal(Signal(name='behavior_condition',
                                   matrix=m['behavior_condition'],
                                   fs=self.meta['respfs']))

    def add_signal(self, sig):
        if not type(sig) == Signal:
            raise ValueError("Signals must be of type Signal()")
        self.signals.append(sig)

    def savetodir(self, dirpath=DEFAULT_DIRPATH):
        for s in self.signals:
            s.savetocsv(dirpath)


def mat2recordings(matfile):
    """ Converts a matlab file into multiple Recordings """
    matdata = scipy.io.loadmat(matfile,
                               chars_as_strings=True,
                               squeeze_me=False)

    # Verify that .mat file has no unexpected matrix variables
    expected_matlab_vars = set(['data', '__globals__', '__version__',
                                '__header__', 'cellid'])
    found_matrices = set(matdata.keys())
    if not found_matrices == expected_matlab_vars:
        raise ValueError("Unexpected variables found in .mat file: "
                         + found_matrices)

    recordings = [MatlabRecording(matdata=r) for r in matdata['data'][0, :]]

    return recordings


def extract_metadata(matrec):
    """ Extracts metadata from the matlab matrix object matrec. """
    found = set(matrec.dtype.names)
    needed = ['cellid', 'isolation', 'stimfs', 'respfs',
              'stimchancount', 'stimfmt', 'filestate']
    unwrap = lambda n: n[0] if type(n) == np.ndarray else n
    meta = dict((n, unwrap(matrec[n][0])) for n in needed if n in found)

    meta['stimparam'] = [str(''.join(letter)) for letter in matrec['fn_param']]

    # Tags is not completely examined here; TODO: find others?
    meta['prestim'] = matrec['tags'][0]['PreStimSilence'][0][0][0]
    meta['poststim'] = matrec['tags'][0]['PostStimSilence'][0][0][0]
    meta['duration'] = matrec['tags'][0]['Duration'][0][0][0]

    # TODO: 'fn_spike', 'fn_param' metadata is ignored; too complicated!

    # TODO: I don't think these should be metadata. Remove anywhere found.
    # meta['est'] = matrec['estfile']
    # meta['repcount']=np.sum(np.isfinite(data['resp'][0,:,:]),axis=0)

    return meta


###############################################################################
# SCRIPT STARTS HERE

matfile = ('/home/ivar/tmp/gus027b-a1_b293_parm_fs200'
           + '/gus027b-a1_b293_parm_fs200.mat')

recordings = mat2recordings(matfile)

for r in recordings:
    for s in r.signals:
        print("shape of", s.name,  s.__matrix__.shape)
        print("avg of", s.name,  s.as_average_trial().shape)
        print("app of", s.name,  s.as_time_stream().shape)

    # r.savetodir('/home/ivar/mat/')

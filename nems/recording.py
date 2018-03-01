import io
import os
import gzip
import time
import tarfile
import logging
import pandas as pd
import numpy as np
import copy
import nems.epoch as ep
from .signal import Signal

log = logging.getLogger(__name__)


class Recording:

    def __init__(self, signals):
        '''
        Signals argument should be a dictionary of signal objects.
        '''
        self.signals = signals

        # Verify that all signals are from the same recording
        recordings = [s.recording for s in self.signals.values()]
        if not recordings:
            raise ValueError('A recording must contain at least 1 signal')
        if not len(set(recordings)) == 1:
            raise ValueError('Not all signals are from the same recording.')

        self.name = recordings[0]

    def copy(self):
        '''
        Returns a copy of this recording.
        '''
        return copy.copy(self)

    @property
    def epochs(self):
        '''
        The epochs of a recording is the superset of all signal epochs.
        '''
        return pd.concat([s.epochs for s in self.signals.values()])

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
    def load(directory_or_targz):
        '''
        Loads all the signals (CSV/JSON pairs) found in DIRECTORY or
        .tar.gz file, and returns a Recording object containing all of them.
        '''
        if os.path.isdir(directory_or_targz):
            files = Signal.list_signals(directory_or_targz)
            basepaths = [os.path.join(directory_or_targz, f) for f in files]
            signals = [Signal.load(f) for f in basepaths]
            signals_dict = {s.name: s for s in signals}
            return Recording(signals=signals_dict)
        elif os.path.exists(directory_or_targz):
            with open(directory_or_targz, 'rb') as stream:
                return Recording.load_from_targz_stream(stream)
        else:
            m = 'Not a dir or .tar.gz file: {}'.format(directory_or_targz)
            raise ValueError(m)

    @staticmethod
    def load_from_targz_stream(tgz_stream):
        '''
        Loads the recording object from the given .tar.gz stream, which
        is expected to be a io.BytesIO object.
        '''
        streams = {}  # For holding file streams as we unpack
        with tarfile.open(fileobj=tgz_stream, mode='r:gz') as t:
            for member in t.getmembers():
                basename = os.path.basename(member.name)
                # Now put it in a subdict so we can find it again
                signame = str(basename.split('.')[0:2])
                if basename.endswith('epoch.csv'):
                    keyname = 'epoch_stream'
                elif basename.endswith('.csv'):
                    keyname = 'csv_stream'
                elif basename.endswith('.json'):
                    keyname = 'json_stream'
                else:
                    m = 'Unknown file situation: {}'.format(member.name)
                    raise ValueError(m)
                # Ensure that we can doubly nest the streams dict
                if signame not in streams:
                    streams[signame] = {}
                # Read out a stringIO object for each file now while it's open
                f = io.StringIO(t.extractfile(member).read().decode('utf-8'))
                streams[signame][keyname] = f
        # Now that the streams are organized, convert them into signals
        log.debug({k: streams[k].keys() for k in streams})
        signals = [Signal.load_from_streams(**sg) for sg in streams.values()]
        signals_dict = {s.name: s for s in signals}
        return Recording(signals=signals_dict)

    def save(self, directory, no_subdir=False, compressed=False):
        '''
        Saves all the signals (CSV/JSON pairs) in this recording into
        DIRECTORY in a new directory named the same as this recording.
        If optional argument "compressed" is True, it will save the
        recording as a .tar.gz file instead of as files in a directory.
        '''
        if compressed:
            filename = self.name + '.tar.gz'
            filepath = os.path.join(directory, filename)
            with open(filepath, 'wb') as archive:
                tgz = self.as_targz()
                archive.write(tgz.read())
                tgz.close()
        else:
            directory = os.path.join(directory, self.name)
            if not os.path.isdir(directory):
                os.makedirs(directory)
            for s in self.signals.values():
                s.save(directory)

    def as_targz(self):
        '''
        Returns a BytesIO containing all the rec's signals as a .tar.gz stream.
        You may either send this over HTTP or save it to a file. No temporary
        files are created in the creation of this .tar.gz stream.

        Example of saving an in-memory recording to disk:
            rec = Recording(...)
            with open('/some/path/test.tar.gz', 'wb') as fh:
                tgz = rec.as_targz()
                fh.write(tgz.read())
                tgz.close()  # Don't forget to close it!
        '''
        f = io.BytesIO()  # Create a buffer
        tar = tarfile.open(fileobj=f, mode='w:gz')
        # tar = tarfile.open('/home/ivar/poopy.tar.gz', mode='w:gz')
        # With the tar buffer open, write all signal files
        for s in self.signals.values():
            d = s.as_file_streams()  # Dict mapping filenames to streams
            for filename, stringstream in d.items():
                if type(stringstream) is io.BytesIO:
                    stream = stringstream
                else:
                    stream = io.BytesIO(stringstream.getvalue().encode())
                info = tarfile.TarInfo(os.path.join(self.name, filename))
                info.uname = 'nems'  # User name
                info.gname = 'users'  # Group name
                info.mtime = time.time()
                info.size = stream.getbuffer().nbytes
                tar.addfile(info, stream)
        tar.close()
        f.seek(0)
        return f

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

    def _split_helper(self, fn):
        '''
        For internal use only by the split_* functions.
        '''
        est = {}
        val = {}
        for s in self.signals.values():
            (e, v) = fn(s)
            est[e.name] = e
            val[v.name] = v
        return (Recording(signals=est), Recording(signals=val))

    def split_at_time(self, fraction):
        '''
        Calls .split_at_time() on all signal objects in this recording.
        For example, fraction = 0.8 will result in two recordings,
        with 80% of the data in the left, and 20% of the data in
        the right signal. Useful for making est/val data splits, or
        truncating the beginning or end of a data set.
        '''
        return self._split_helper(lambda s: s.split_at_time(fraction))

    def split_by_epochs(self, epochs_for_est, epochs_for_val):
        '''
        Returns a tuple of estimation and validation data splits: (est, val).
        Arguments should be lists of epochs that define the estimation and
        validation sets. Both est and val will have non-matching data NaN'd out.
        '''
        return self._split_helper(lambda s: s.split_by_epochs(epochs_for_est,
                                                              epochs_for_val))

    def split_using_epoch_occurrence_counts(self, epoch_regex):
        '''
        Returns (est, val) given a recording rec, a signal name 'stim_name', and an 
        epoch_regex that matches 'various' epochs. This function will throw an exception
        when there are not exactly two values for the number of epoch occurrences; i.e.
        low-rep epochs and high-rep epochs. 
        
        NOTE: This is a fairly specialized function that we use in the LBHB lab. We have 
        found that, given a limited recording time, it is advantageous to have a variety of sounds
        presented to the neuron (i.e. many low-repetition stimuli) for accurate estimation
        of its parameters. However, during the validation process, it helps to have many
        repetitions of the same stimuli so that we can more accurately estimate the peri-
        stimulus time histogram (PSTH). This function tries to split the data into those
        two data sets based on the epoch occurrence counts.
        '''
        groups = ep.group_epochs_by_occurrence_counts(self.epochs, epoch_regex)
        if len(groups) > 2:
            l=np.array(list(groups.keys()))
            k=l>np.mean(l)
            hi=np.max(l[k])
            lo=np.min(l[k==False])
            
            g={hi: [], lo: []}
            for i in list(np.where(k)[0]):
                g[hi]=g[hi]+groups[l[i]]
            for i in list(np.where(k==False)[0]):
                g[lo]=g[lo]+groups[l[i]]
        elif len(groups)<2:
            m = "Fewer than two types of occurrences (low and hi rep). Unable to split:"
            m += str(groups)
            raise ValueError(m)
                
        n_occurrences = sorted(groups.keys())
        lo_rep_epochs = groups[n_occurrences[0]]
        hi_rep_epochs = groups[n_occurrences[1]]
        return self.split_by_epochs(lo_rep_epochs, hi_rep_epochs)

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
                          invert=False, excise=False):
        '''
        By default, calls jackknifed_by_time on all signals and returns a new
        set of data.  If you would only like to jackknife certain signals,
        while copying all other signals intact, provide their names in a
        list to optional argument 'only_signals'.
        '''
        if excise and only_signals:
            raise Exception('Excising only some signals makes signals ragged!')
        new_sigs = {}
        for sn in self.signals.keys():
            if (not only_signals or sn in set(only_signals)):
                s = self.signals[sn]
                new_sigs[sn] = s.jackknife_by_time(nsplits, split_idx,
                                                   invert=invert, excise=excise)
        return Recording(signals=new_sigs)

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

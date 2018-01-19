'''
In contrast to the current format used by NEMS where data is in the shape trial
x channel x time, Signal objects treat the data as a single long timeseries of
channel x time. Signal objects also contain information about when trials
began/ended, allowing us to extract trials or average across trials. However,
trial-based operations should be restricted to plotting purposes. All module
operations in the stack should operate on the full timeseries itself rather than
on a per-trial basis.
'''
import numpy as np
import pandas as pd

from nems import db
from nems.utilities import baphy
from nems.utilities import io
from nems.data.api import Recording, Signal


def _load_row(row, fs):
    # Using MATLAB-like syntax, construct the options that will be used to
    # determine the actual filenames that contain the data we need.
    stim_options = {
        'filtfmt': 'ozgf',
        'fsout': fs,
        'chancount': 18,
        'includeprestim': 1
    }
    stimfile = baphy.stim_cache_filename(row['stim'], stim_options)

    resp_options = {'rasterfs': fs, 'includeprestim': 1}
    respfile = baphy.spike_cache_filename2(row['raster'], resp_options)

    # Using the filenames returned by the helper functions, load the data we
    # need (note that stim1 and stim2 are special cases for Brad's data). In
    # theory we should be able to do all of this in a for loop through the items
    # in the row (and store the results in a dictionary) rather than pulling in
    # each dataset separately; however, each dataset has slightly different
    # arguments for loading (e.g., the ordering of the data may be different in
    # the file).
    channel_ordering = dict(channelaxis=0, eventaxis=2, timeaxis=1)
    stim1 = io.load_matlab_matrix(stimfile, 'stim1', **channel_ordering)
    stim2 = io.load_matlab_matrix(stimfile, 'stim2', **channel_ordering)

    channel_ordering = dict(channelaxis=1, eventaxis=2, timeaxis=0)
    phase = io.load_matlab_matrix(stimfile, 'repeating_phase', **channel_ordering)
    target = io.load_matlab_matrix(stimfile, 'targetid', **channel_ordering)
    stream = io.load_matlab_matrix(stimfile, 'singlestream', **channel_ordering)

    resp = io.load_matlab_matrix(respfile, 'r', repaxis=1, eventaxis=2, timeaxis=0)

    # Build a dataframe that contains the indices of the trial starts and ends.
    # This is necessary because the internal representation used by the
    # Signal/Recording objects is a single long timeseries (shape n_channels x
    # n_timepoints). The trial_info is used to convert this timeseries back into
    # a 3D array of (n_trials, n_channels, n_times) when needed. Right now the
    # data is in the format (n_chans, n_trials, n_time). When creating the
    # Signal/Recording objects, this will be reshaped into (n_chans x n_time).
    n_chans, n_trials, n_time = stim1.shape

    # I'm not making any special provision here to account for the prestim and
    # poststim silence. We could certainly do that at some point.
    trial_info = pd.DataFrame({
        'start_index': np.arange(n_trials)*n_time,
        'end_index': np.arange(n_trials)*n_time + n_time,
    })

    # Build a dictionary so we can loop through all of our data more quickly in
    # the next few lines of code.
    data = {
        'stim1': stim1,
        'stim2': stim2,
        'phase': phase,
        'target': target,
        'stream': stream,
        'resp': resp,
    }

    # Make sure that all datasets have the same number of trials. If not,
    # something went wrong.
    n_trials = [v.shape[1] for v in data.values()]
    if len(set(n_trials)) != 1:
        raise ValueError("Data does not have same number of events")

    # Now, reshape each dataset into a long timeseries and createa Signal object
    # from it. The set of signals will be saved as a recording.
    signals = {}
    for k, v in data.items():
        matrix = v.reshape((len(v), -1)).astype(np.float)
        chans = [''.format(c) for c in np.arange(len(matrix))]
        signals[k] = Signal(100, matrix, k, cellid, chans=chans,
                            trial_info=trial_info)
    recording = Recording(signals)

    return recording


def load_data(data, fs=100):
    # Load each set of files and concatenate them all together into one long
    # recording.
    recordings = [_load_row(row, fs) for _, row in data.iterrows()]
    return Recording.concatenate_recordings(recordings)


if __name__ == '__main__':
    cellid = 'sti016a-b1'
    batch = 269

    # This returns a dataframe. Each row represents a file from that dataframe
    # which will be merged into a single recording (information about trials are
    # preserved so we can properly refactor the trials).
    data = db.get_batch_cell_data(batch, cellid)

    # This loads the data in each file and merges it into a single recording.
    recording = load_data(data)

    # Note that this doesn't actually do the preferred approach of pulling out
    # validation datasets across the full recording yet. This will be implemented in
    # the near-future. The point is that the recording knows how to split the
    # dataset into estimation and validation sets.
    est = recording.jackknifed_by_time(10, 0, invert=False)
    val = recording.jackknifed_by_time(10, 0, invert=True)

    e = np.nanmean(est.signals['stim1']._matrix, axis=0)
    v = np.nanmean(val.signals['stim1']._matrix, axis=0)

    # Number of timepoints that are valid
    print(len(e[~np.isnan(e)]))
    print(len(v[~np.isnan(v)]))

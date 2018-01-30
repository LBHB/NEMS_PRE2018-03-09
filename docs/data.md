Recording
=========
A `Recording` is a collection of signals that were recorded simultaneously,
such that their time indexes line up (and thus the time dimension of all
signals should be the same length and have the same sampling rate).

Signals
=======
A `Signal` is a convenience class for slicing, averaging, jackknifing,
truncating, splitting, saving, and loading tabular data that is stored in a CSV
file + a JSON metadata sidecar. This class is intended to be useful for loading
a dataset, dividing it into estimation and validation subsets, selecting parts
of data where a condition is true, concatenating Signals together, and other
common data wrangling tasks.

Data access 
...........

Generally speaking, you won't access a Signal's internal ._matrix data directly
(it is immutable anyway), but instead call a function that will present the
data to you in the format you want. For example:

    sig = nems.signal.load('/path/to/signal')
    data = sig.as_single_trial()
    # or
    data = sig.as_repetition_matrix()
    # or
    data = sig.as_average_trial()

Transforming a signal
.....................

It's very common to want to create a new signal from an existing signal.  You
may do that with the following functions:

    .normalized_by_bounds()
    .normalized_by_mean()
    .split_by_reps(fraction)
    .split_by_time(fraction)
    .jackknifed_by_reps(nsplits, split_idx)
    .jackknifed_by_time(nsplits, split_idx)

Combining signals
.................

Other common operations are to combine another signal with this one:

    .append_signal(other_signal)
    .combine_channels(other_signal)

Selecting subsets
.................

Another really common operation is to want to select just part of this signal,
and NaN out the rest. This is accomplished by using the ".segments"
annotations, which give names to the start and stop time indexes of various
intervals of this signal. For example:

    sig = Signal(name='stim', ...)
    sig.segments = [['TORC1', 0, 500 ], # Torc 1 is from time 0 to 500
                    ['TORC2', 501, 1000 ],
                    ['TORC1', 1001, 1500 ]]) # Another rep of TORC1 is ok
    gain1 = np.nanmean(sig.select('TORC1'))
    gain2 = np.nanmean(sig.select('TORC2'))
    def some_scaling_function(x):
    ... TODO ...
    newsig = sig.apply('TORC1', some_scaling_function)

File format
...........

A CSV file should have one row per instant in time, and each column should be a
different "channel" of the signal. Channels can represent whatever dimension
you want, such as an auditory frequency, X/Y/Z coordinates for motion, or
voltage and current levels for a neuron.  It's common for Signals to have
multiple channels, because it is common for a tuple of data to be measured at
the same instant.

The JSON file specifies optional attributes for the Signal, such as:

    .name       The name of the signal
    .recording  The name of the recording session of this signal
    .chans      A list of the names of each of the channels
    .fs         Frequency of sampling [Hz]
    .segments   Named tuplets that annotate a time interval
    .meta       A catch-all data structure for anything else you want

You may augment the .meta with whatever information describes the experimental
conditions under which that the data was observed.

Operations
..........

Signals implement the Numpy universal function interface. This means that you
can perform a variety of array operations on Signals:

    # Add a DC offset of 5 to the signal
    offset_signal = signal + 5

    # Matrix multiplication
    weighted_channels = weights @ signal

    # Multi-signal operations (stim and pupil are signals)
    pred = stim * pupil + stim * pupil**2 + stim * pupil**3

    # Apply a linear filter to the signal. A new signal is created as fir
    fir = lfilter(b, a, stim)

    # Now, average across the filtered channels.
    fir_mean = fir.mean(axis=0)

When performing an operation on a signal, a new signal object is returned. The
signal will be identical to the original object, albeit with
appropriately-transformed data (e.g., sampling rate and epochs will be copied
over). If you attempt to perform an operation (e.g., adding two signals) that
do not match in some attribute (e.g., number of samples, sampling rate, etc.)
you'll get an error.

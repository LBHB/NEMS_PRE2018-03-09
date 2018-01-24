## Specs for Recording/Signal objects

SVD 2018-01-23

## What is a recording?

A collection of *signals* collected over a period of time 1...T and an *epochs* DataFrame that identifies important time segments in the experiment

## What does it look like on disk?

For example:

```
└── gus027b13_p_PPS
   ├── gus027b13_p_PPS.pupil.csv
   ├── gus027b13_p_PPS.pupil.epochs.csv
   ├── gus027b13_p_PPS.pupil.json
   ├── gus027b13_p_PPS.resp.csv
   ├── gus027b13_p_PPS.resp.epochs.csv
   ├── gus027b13_p_PPS.resp.json
   ├── gus027b13_p_PPS.stim.csv
   ├── gus027b13_p_PPS.stim.epochs.csv
   └── gus027b13_p_PPS.stim.json 
```

As you can see, there is one directory per recording, and for each signal there are two tabular files ('epochs' is a 3 column CSV containing events and epochs) and one JSON. Depending on the signal type, there might not be an epoch file at all.

## What is a signal?

A signal can come in a few different forms:

1. A continuous recording of N channels. This can be represented as an N x T array. This might represent pupil, LFP, or a continues  measure of spike rate at each point in time.

2. (NEW?) A list of discrete event times. Eg, the time of an action potential or lick. Say you have M different categories of discrete events. Then you need M lists of times ranging 0 ... T. These could be converted into an M x T matrix, but again, that might wait until nems actually needs it. This is actually how we store spike data in baphy.
   ```discrete_times['lick']=[0.43, 1.3, 1.4, 1.55, ... ]   # list of times when the animal licked
   discrete_times['spk1']=[0.004, 0.533, 1.434, .... ] # list of times when neuron 1 spiked
   ```
   
3. (NEW?) Event signal: The same event might occur many times during an experiment (eg, a stimulus). Rather than storing a continuous recording of events, it seems like it would make sense to store one copy of each unique event and convert into a continuous trace when nems needs it. Say we have a P-channel spectrogram and 30 different stimulus events with maxium duration S.  The event signal can be stored in a [P x 30 x S] array.  A list of 30 string provide a label for each unique event.

## What does the *epochs* DataFrame look like?

It is a simple table consisting of one row per specified time segment,
with three columns: start_index, end_index, and epoch_name.
An example signal.epochs DataFrame might look like:

```
   start_index   end_index   epoch_name
0            0         100       TORC01
1          100         200       TORC02
2          120         160    MY_EPOCH7
3          180         380       TORC03
```

Indices behave similar to python list indexes,
so start times are inclusive whereas end times are exclusive.
Time segments can overlap, can have the same name, etc.
No rules are enforced in regards to formatting.

An *epochs* DataFrame must be specified in order to use
some of the signal methods, such as split_at_epoch and fold_by.
For simple signals with predictable patterns,
some methods will be provided for generating generic
a generic *epochs* DataFrame (ex: 10 non-overlapping
trial or stimulus segments of equal length).
For more complex signals, users will need to
either preprocess the epochs information accordingly
or specify it manually after loading the signal.

Ex:

```
sig = signal.load('/path/to/signal')
my_epochs = pandas.DataFrame({'start_index': [0, 10, 20],
                              'end_index': [10, 20, 30],
                              'epoch_name': ['one', 'two', 'three']},
                              columns=['start_index', 'end_index', 'epoch_name'])
sig.epochs = my_epochs
sig.fold_by('one')
# Would match the epoch 'one' and return a 1 x num_channels x 10 ndarray
# containing the data from time bins 0 through 10
```

NOTE: final structure/format of the *epochs* information
      is still a work-in-progress, as are the naming and
      behavior of the epochs-related signal methods.


## What do we need to be able to do?

1. select a subset of events from the event list based on some string processing. Simplest case: find every occurences of "TORCnn" in the event list. assign a unique eventid=nn to each distinct "TORCnn". Figure out what to do with pre- and post-stim silences based on [smart way of representing events]. Use that to generate the list of startime and stoptime for each matching event. The result is a list of eventid and epochs, where the value of eventid ranges from 1...30 in the case of TORCs. And if the TORCs were repeated 5 times, the event list should have 150 entries:
```
[eventid1 starttime1 stoptime1
 eventid2 starttime2 stoptime2
 ....
]
```

Example for case 1 using the current *epochs* implementation,
where sig._matrix is a 3 channel x 900 time bin ndarray
(Going with the description above of 30 TORCS repeated 5 times each
 and deciding arbitrarily that they occupy 3s each. Apologies
 if the numbers are nonsense but it should get the idea across):

```
>>> sig.epochs
   start_index   end_index   epoch_name
0            0           6       TORC00  #first rep
1            6          12       TORC01
...
29         174         180       TORC29
30         180         186       TORC00  #second rep
31         186         192       TORC01
...
...
...
149        894         900       TORC29  #end of fifth rep

# Now we want to look at the data for TORC01 only:
# NOTE: simple select method not actually implemented yet
#       in dev branch, but I think bburan had something in RDT?
>>>sig.select('TORC01')
# Returns a 3 channels x 900 time bin ndarray with
# all values set to NaN except for bins 6-12, 186-192,
# 366-372, 546-552, and 726-732, corresponding to
# the five repetitions.

# Now we want to reshape the data to index by the
# five reptitions of TORC29 as the first dimension.
>>>sig.fold_by('TORC29')
# returns a 5 epochs x 3 channels x 6 time bins ndarray,
# where array[0] corresponds to the first repetition of
# TORC29, array[1] the second repetition, etc.
# The length of the time axis is determined by the
# longest epoch matched (in this case all are length 6),
# and shorter-length matches are appended with NaN values.

# NOTE: the epochs selection methods accept
#       regular expressions as arguments, not just strings.
#       So more complex selections can be composed by
#       taking advantage of that syntax. For example:
>>>sig.fold_by('^TORC(00|01|02)$') # to match all reps of the first three

```

2. using the event list:

  a. extract a raster from the continuous recordings [N x unique-event-count X max-event-duration]

  b. generate a raster from the discrete event times: [M x unique-event-count X max-event-duration]
 
  c. contruct an event-signal matrix from the event signals: [P x unique-event-count X max-event-duration]






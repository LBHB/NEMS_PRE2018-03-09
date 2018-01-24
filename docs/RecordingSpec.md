## Specs for Recording/Signal objects

SVD 2018-01-23

## What is a recording?

A collection of *signals* collected over a period of time 1...T and an *epochs* DataFrame that identifies important time segments in the experiment

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

2. using the event list:

  a. extract a raster from the continuous recordings [N x unique-event-count X max-event-duration]

  b. generate a raster from the discrete event times: [M x unique-event-count X max-event-duration]
 
  c. contruct an event-signal matrix from the event signals: [P x unique-event-count X max-event-duration]






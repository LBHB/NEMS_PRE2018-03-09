## Specs for Recording/Signal objects

SVD 2018-01-23

## What is a recording?

A collection of *signals* collected over a period of time 1...T and an *event list* that identifies important epochs in the experiment

## What is a signal?

A signal can come in a few different forms:

1. A continuous recording of N channels. This can be represented as an N x T array. This might represent pupil, LFP, or a continues  measure of spike rate at each point in time.

2. (NEW?) A list of discrete event times. Eg, the time of an action potential or lick. Say you have M different categories of discrete events. Then you need M lists of times ranging 0 ... T. These could be converted into an M x T matrix, but again, that might wait until nems actually needs it. This is actually how we store spike data in baphy.

3. (NEW?) Event signal: The same event might occur many times during an experiment (eg, a stimulus). Rather than storing a continuous recording of events, it seems like it would make sense to store one copy of each unique event and convert into a continuous trace when nems needs it. Say we have a P-channel spectrogram and 30 different stimulus events with maxium duration S.  The event signal can be stored in a [P x 30 x S] array.  A list of 30 string provide a label for each unique event.

## What is an event list?

It should be simple. 
```
[starttime stoptime string
 ...
]
```
Presumably the string should be meaningful, but I would start by just using the Note field from baphy parameter files. For W events, we have a 3 X W list:


## What do we need to be able to do?

1. select a subset of events from the event list based on some string processing. Simplest case: find every occurences of "TORCnn" in
the event list. assign a unique eventid=nn to each distinct "TORCnn". Figure out what to do with pre- and post-stim silences based
on <smart way of representing events>. Use that to generate the list of startime and stoptime for each matching event. The result is a list of eventid and epochs, where the value of eventid ranges from 1...30 in the case of TORCs. And if the TORCs were repeated 5 times, the event list should have 150 entries:
```
[eventid1 starttime1 stoptime1
 eventid2 starttime2 stoptime2
 ....
]
```

2. using the event list:

  2a. extract a raster from the continuous recordings [N x unique-event-count X max-event-duration]

  2b. generate a raster from the discrete event times: [M x unique-event-count X max-event-duration]
 
  2c. contruct an event-signal matrix from the event signals: [P x unique-event-count X max-event-duration]






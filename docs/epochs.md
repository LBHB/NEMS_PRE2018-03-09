# Epochs #

## What's an Epoch?

An epoch is an event or period of time that has been tagged with a name. Epochs are stored in Signal objects (NOT in a Recording object, although Recordings have methods that wrap and use the superset of all of their signals' epochs), and are represented as triplets:

```
(start_index, end_index, epoch_name)
```

where `start_time` and `end_time` are the moments of time marking the beginning and end of the epoch, and `epoch_name` is a string that tags this period of time. You can tag multiple epochs with the same `epoch_name` as you wish; there are other ways for us to select them. 

Zero-length (start_index == end_index) epochs are allowed. 

```
TODO: As of 2018-01-24, epochs use indexes instead of absolute time.
In the future, we should switch to using absolute time.
This means that start_index, start_index should become start_time, end_time.
If we do this, resampling frequencies and merging signals is never problematic.
```

## What do Epochs Look Like in Practice?

For the rest of this document, let's use example epoch data in which trials are 40-100 units long, and there are several 20-unit long TORCs played before a tone-in-TORC detection task.

```
0, 60, ExperimentalTrial
0, 5, PreStimSilence
0, 20, TORC_3983
0, 20, Reference
15, 20, PostStimSilence
20, 25, PreStimSilence
20, 40, TORC_572
20, 40, Reference
35, 40, PostStimSilence
40, 45, PreStimSilence
40, 60, TORC_3983
40, 60, PureTone12
40, 60, DetectionTask
49, 49, Licking
55, 60, PostStimSilence
60, 100, ExperimentalTrial
60, 65, PreStimSilence
60, 80, TORC_444
60, 80, Reference
66, 87, Licking
75, 80, PostStimSilence
80, 100, TimeOut
```

As we can see, many of the epochs overlap in interesting ways. In the first experimental trial two reference torcs are played (TORC3983, TORC572), then a tone-in-torc is played (The simultaneous TORC_3983, PureTone12), and the animal correctly licks. In the second experimental trial, the animal licks during the reference sound and gets a time out. 

We will now ask some simple questions and show example operations that we might do on this data.

### How do we pick out only the trials in which the animal licks correctly?

When fitting the data, we might want to take only 'correct' trials in which the animal licks during a DetectionTask. We could do this using `overlapping_epochs`, which returns the outer bounds of any region of time in which two epochs overlap. Mostly, this is useful for long events (like trials) and detecting when one or more short events (like licks, or correct behavior events, or whatever) occur inside the long event. TODO: I don't like the name 'overlapping_epochs'! What is this operation really called?!)

We also need to use the `.select_epochs()` function, which NaNs everything but the selected epochs.

```
correct_detections = signal.overlapping_epochs('DetectionTask', 'Licking')
correct_trials = signal.overlapping_epochs('ExperimentalTrial', correct_detections)
data = signal.select_epochs[correct_trials]
```

Great! You can save that for later by adding it to the epochs:

```
# If we permanently add the correct_trials epochs...
signal.add_epochs('CorrectTrial', correct_trials)

# ...then anytime afterward we can simply do
data = signal.select_epochs('CorrectTrial')
```

### How do you remove the prestim and poststim silence?

Besides `.overlapping_epochs()`, you may also use set theory (or Boolean Logic, if you prefer):

```
# Get only the prestim silence by combining using an INTERSECTION (AND) operation
only_prestim = signal.combine_epochs('CorrectTrial', 'PreStimSilence', op='intersection')
data = signal.select_epochs(only_prestim)

# Remove the prestim silence by combining using a DIFFERENCE (XOR) operation
no_prestim = signal.combine_epochs('CorrectTrial', 'PreStimSilence', op='difference')
```


### How do I get the average response to a particular TORC?

Rather than just NaNing data with `signal.select_epochs()`, you may also create a 3D array by making a 2D array for every occurence of a particular epoch_name.

```
data = signal.fold_by('TORC_3983')
average_response = data.mean(axis=0)
```

Note that the shape of `data` is `(O, C, T)`, where `O` indexes the occurrences of the 'TORC_3983' epochs (2 in this case), `C` is an index into channels, and `T` is an index into time. Thus `average_response` must have shape `(C, T)`. 


### How do I get the average response in prestim vs poststim, regardless of behavior?

This might be useful for identifying a baseline that is altered by behavior.

```
prestim = signal.select_epochs('PreStimSilence').mean(axis=1)
poststim = signal.select_epochs('PostStimSilence').mean(axis=1)
```

### How do I get the average stimulus 300ms before every mistaken lick?

What if we want to know what the animal heard just before it licked accidentally? Or if the TORC was maybe too close to the reference tone?

```
# Identify the bad trials
bad_trials = signal.combine_epochs('Trials', 'CorrectTrials', op='difference')

# Extend the 'licking' events backward 300ms
prior_to_licking = signal.extend_epoch('Licking', 300, 0)

# Now take the intersection of those two selections
before_bad_licks = signal.combine_epochs(bad_trials, prior_to_licking, op='intersection')
some_plot_function(signal.select_epochs(before_bad_licks))

```

Note that fold_by may end up duplicating data. For example, if the animal licked 10 times a second and you were looking at the 3 seconds prior to each lick, you just duplicated your data almost 30 times! This may negatively alter certain computations of the mean in some sense, and in such circumstances, you may want to use the argument `allow_data_duplication=False` for `signal.fold_by()`.

TODO: Note that we have focused on single signals here. In practice, data selection will be very slightly more difficult than this because I'm not sure how stimulus data is actually represented in Baphy. 'stimTORCs' might be one signal, and 'stimTones' might be another signal, or they might be combined into a multi-channel signal, or something entirely different.

### How do I use epoch info from two different signals in the same recording?

Instead of using Signal's `.combine_epochs()`, `.select_epochs()`, and `.fold_by()` methods, use the corresponding ones in the Recording class. (TODO)

## How should I name epochs?

Be descriptive. If you give a stimulus a unique name, then when it occurs in other Recordings,  you can simply concatenate the two recordings and still select exactly the same data.

Avoid implicit indexes like `trial1`, `trial2`, `trial3`; prefer using just `trial` and the folding functionality of `.fold_by('trial')`, which gives you a matrix. If you have truly different stimuli, you may named them `stim01`, `stim02`, but descriptive names like `cookoo_bird.wav`, and `train_horn.wav` are better.

Remember that the idea behind epochs is to tag the content of data, much like HTML marks up text to tell what it is. It's totally fine to tag the exact same epoch with multiple names, if that will help you perform queries on it later.

## What happens with zero-length epochs?

TODO: It is untested, but I think they work fine with `.overlapping_epochs()`. They will not work at all with `.combine_epochs()`, because that is an impossible operation.

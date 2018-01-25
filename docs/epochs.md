# Epochs

## What's an Epoch?

An epoch is an event or period of time that has been tagged with a name. Epochs
are attached to Signal objects. A Recording, which contains a collection of
signals, has methods that allow us to manipulate the entire collection based on
a superset of the epochs in each signal.

A single epoch has three values::

	start_time, end_time, epoch_name

where `start_time` and `end_time` denote the beginning and end of the epoch in
seconds and `epoch_name` is a descriptive string (see [how to name epochs](##
Epoch naming). For events (single points in time), the `start_time` will
contain the timestamp of the event and `end_time` will be NaN.

Multiple epochs are stored in a pandas DataFrame containing three columns
(`start_time`, `end_time` and `epoch_name`).

For the rest of this document, let's use example epoch data in which trials are
40-100 msec long, and there are several TORCs of 20 msec duration played before
a tone-in-TORC detection task:

	start_time  end_time  	epoch_name

	0.00  		0.060   	ExperimentalTrial
	0.00  		0.05  		PreStimSilence
	0.00  		0.020  		TORC_3983
	0.00  		0.020  		Reference
	0.015   	0.020  		PostStimSilence
	0.020   	0.025  		PreStimSilence
	0.020   	0.040  		TORC_572
	0.020   	0.040  		Reference
	0.035   	0.040  		PostStimSilence
	0.040   	0.045  		PreStimSilence
	0.040   	0.060  		TORC_3983
	0.040   	0.060  		PureTone12
	0.040   	0.060  		DetectionTask
	0.049   	0.049  		Licking
	0.055   	0.060  		PostStimSilence
	0.060   	0.0100  	ExperimentalTrial
	0.060   	0.065  		PreStimSilence
	0.060   	0.080  		TORC_444
	0.060   	0.080  		Reference
	0.066   	0.087  		Licking
	0.075   	0.080  		PostStimSilence
	0.080   	0.0100  	TimeOut

In the example above, note that some epochs are duplicated but with different
names. For example, compare the first `TORC_3983` with the first `Reference`.
This is a way of indicating that `TORC_3983` is a `Reference` token. This
approach facilitates analysis where one may wish to select all reference TORCs
and compare them to all TORCs that occur simultaneously with a pure tone
(compare the second occurence of `TORC_3983` with `PureTone12`).

This set of epochs tells us quite a bit about what's going on in the
experiment. In the first trial two reference TORCs are played (`TORC_3983`,
`TORC_572`), then a tone-in-TORC is played (`TORC_3983`, `PureTone12`), and the
animal correctly licks. In the second trial, the animal licks during the
reference sound and gets a time out.

## How signals use epochs

A signal object can use epochs to perform two basic operations:

* Mask regions of data. For example, perhaps licking introduces EMG artifacts
  in to the LFP recordings. In this case, you may want to mask all regions in
  the LFP recording during a lick so that your analysis isn't affected by these
  artifacts:

  	 signal.mask_epochs('Licking', inplace=True)

  As you will see later, this masking can also be used to generate subsets of
  data for cross-validation when fitting models. Signals also have a
  `select_epochs` method, which is the inverse of `mask_epochs`:

  	 signal.select_epochs('Reference', inplace=True)

  **Brad's comment - We need to think about what happens when we mask and/or
  select epochs. Are these operations cumulative, or does the selection mask
  reset on each call to `mask_epochs` or `select_epochs`? 
  
  I would argue that the selection mask gets reset each time because it's not
  clear if successive selection operations be OR, AND or XOR (we *could*
  specify a grammar for this or have a method for `select_epochs` and
  `mask_epochs` that indicate whether that operation is OR, AND or XORed to the
  current selection mask, but it gets complicated). Really, if someone needs
  fancy selection behavior, then they should preprocess the epochs first to
  contain the information they need such that a single call to `mask_epochs` or
  `select_epochs` accomplishes what they want).**

* Extract regions of data. For example, perhaps you want to plot the average
  response to a particular epoch:

     all_epochs = signal.extract_epochs('TORC_3983')
	 average_epoch = np.nanmean(all_epochs, axis=0)


## Epoch manipulation

### General epoch selection

When fitting the data, we might want to take only correct trials (defined as
when the animal licks during  in which the animal licks during a DetectionTask.
We could do this using `overlapping_epochs`, which returns the outer bounds of
any region of time in which two epochs overlap. Mostly, this is useful for long
events (like trials) and detecting when one or more short events (like licks,
or correct behavior events, or whatever) occur inside the long event. 

Ivar's comment - I don't like the name 'overlapping_epochs'! What is this
operation really called?

Brad's comment - I feel that we are attempting to do too much here. The
function has to be a bit clever and make some assumptions about what we want. I
would suggest something along the following lines (Note that the epoch
manipulation functions are not Signal methods anymore because they don't really
require a signal object, further, we are no longer *manipulating* epoch
boundaries. If we want to manipulate epoch boundaries, see the next section):

	from nems.data.epochs import select_epochs, epochs_contain

	epochs = signal.get_epochs()

	# Pull out the epochs we want to use in our filter. 
	dt_epochs = select_epochs(epochs, 'DetectionTask')
	l_epochs = select_epochs(epochs, 'Licking')

	# This returns only the set of DetectionTask epochs that contain a Licking
	# epoch that *begins* during the DetectionTask epoch. Mode can be one of
	# {'start', 'end', 'both'}. In the case of 'end', return all dt_epochs that
	# contain the end time for any epoch in l_epochs. In the case of 'both', a
	# full epoch (from l_epoch) must be contained in the dt_epoch for it to be
	# returned.
	correct_epochs = epochs_contain(dt_epochs, l_epochs, mode='start')

Then, we can finally do (to NaN everything but the correct epochs):

	signal.select_epochs(correct_epochs)

Great! You can save that for later by adding it to the epochs in the Signal:

	signal.add_epochs('CorrectTrial', correct_epochs)

Then anytime afterward we can simply do:

	data = signal.select_epochs('CorrectTrial')

### Manipulating epoch boundaries

You can use set theory to manipulate epoch boundaries by subtracting or adding
one epoch to the other:

	from nems.data.epochs import epochs_intersection, epochs_difference

	epochs = signal.get_epochs()
	ct_epochs = select_epochs(epochs, 'CorrectTrial')
	prestim_epochs = select_epochs(epochs, 'PreStimSilence')

	# Get only the prestim silence by combining using an intersection operation
	only_prestim = epochs_intersection(ct_epochs, prestim_epochs)

	# Remove the prestim silence by using a difference operation
	no_prestim = epochs_difference(ct_epochs, prestim_epochs)


### How do I get the average response to a particular epoch?

Instead of masking data with `signal.select_epochs()` and
`signal.mask_epochs()`, you may also extract epochs:

	data = signal.extract_epochs('TORC_3983')
	average_response = np.nanmean(data, axis=0)

Here, `extract_epochs` returns a 3D array with the first axis containing each
occurence of `TORC_3983`. The remaining two axes are channels and time. In this
particular situation, the durations of each occurence of `TORC_3983` are
identical. However, in some situations, the duration of epochs may vary from
occurence to occurence. In this case, shorter epochs will be padded with NaN
values so the length matches the longest occurence. To get the average, use
`np.nanmean`.

### How do I get the average response in prestim vs poststim, regardless of behavior?

This might be useful for identifying a baseline that is altered by behavior.

	signal.select_epochs('PreStimSilence', inplace=True)
	prestim = signal.as_continuous()
	prestim_mean = np.nanmean(prestim)

	signal.select_epochs('PostStimSilence', inplace=True)
	poststim = signal.as_continuous()
	poststim_mean = np.nanmean(poststim)

### How do I get the average stimulus 300ms before every mistaken lick?

What if we want to know what the animal heard just before it licked
accidentally? Or if the TORC was maybe too close to the reference tone?

	# Identify the bad trials
	epochs = signal.get_epochs()

	# Pull out the epochs we want to analyze
	trial_epochs = select_epochs(epochs, 'Trials')
	ct_epochs = select_epochs(epochs, 'CorrectTrials')

	# Note the invert=True. This means to return all trial_epochs that do not
	# contain a ct_epoch.
	bad_trials = epochs_contain(trial_epochs, ct_epochs, invert=True)

	# Extend the 'licking' events backward 300ms
	lick_epochs = select_epochs(epochs, 'Licking')
	prior_to_licking = adjust_epoch(lick_epochs, -300, 0)

	# Now take the intersection of those two selections
	before_bad_licks = epochs_intersection(bad_trials, prior_to_licking)

	signal.select_epochs(before_bad_licks, inplace=True)
	data = signal.as_continous()
	some_plot_function(data)

Note that `extract_epochs` may end up duplicating data. For example, if the
animal licked 10 times a second and you were looking at the 3 seconds prior to
each lick, your data will overlap, meaning you just duplicated your total data
about 1/2 * 3 * 10 = 15 times! This may negatively alter certain computations
of the mean in some sense, and in such circumstances, you may want to use the
argument `allow_data_duplication=False` for `signal.extract_epochs()`.

Ivar's comment: Note that we have focused on single signals here. In practice,
data selection will be very slightly more difficult than this because I'm not
sure how stimulus data is actually represented in Baphy. 'stimTORCs' might be
one signal, and 'stimTones' might be another signal, or they might be combined
into a multi-channel signal, or something entirely different.

### How do I use epoch info from two different signals in the same recording?

Like signal objects, recording objects offer `mask_epochs` and `extract_epochs`
methods. However, you still need to combine the epochs manually. In the above
examples, we assumed that a single signal will contain information about both
the stimulus and whether the animal licked or not. However, that may not always
be the case. Perhaps the "stimulus" signal will contain information about the
stimulus and trials while the "lick" signal will contain information about the
lick epochs (i.e., how the animal responded). For example, if we want to find
anytim the animal blinked or licked and treat those as artifacts and mask the
full recording when they occured).

	lick = recording.get_signal('lick')
	pupil = recording.get_signal('pupil')

	# Note that when we pass a specific epoch name to `get_epochs`, it only 
    # returns the matches rather than the full epochs dataframe.
	blink_epochs = pupil.get_epochs('blinks')
	lick_epochs = lick.get_epochs('Licking')

	all_artifacts = epochs_union(blink_epochs, lick_epochs)
	recording.mask_signals(all_artifacts)

## Epoch naming

Be descriptive. If you give a stimulus a unique name, then when it occurs in
other Recordings,  you can simply concatenate the two recordings and still
select exactly the same data.

Avoid implicit indexes like `trial1`, `trial2`, `trial3`; prefer using just
`trial` and the folding functionality of `.fold_by('trial')`, which gives you a
matrix. If you have truly different stimuli, you may named them `stim01`,
`stim02`, but descriptive names like `cookoo_bird.wav`, and `train_horn.wav`
are better.

Remember that the idea behind epochs is to tag the content of data, much like
HTML marks up text to tell what it is. It's totally fine to tag the exact same
epoch with multiple names, if that will help you perform queries on it later.

## What happens with zero-length epochs?

Zero-length epochs are events. They work best with `epochs_contain`:

	trials = signal.get_epochs('Trial')

	# Assume a laser is an event (i.e., a zero-length epoch)
	laser_pulse = signal.get_epochs('Laser')
	
	laser_trials = epochs_contain(trials, laser_pulse, mode='start')

They will not work with set operations.

## Cross-validation (i.e., jackknifes)

	from nems.data.epochs import jacknife_epochs
	stim = recording.get_signal('stim')
	trials = stim.get_epochs('trials')

	# Generate 20 jacknife sets
	jacknifed_trials = jacknife_epochs(n=20)

	results = []
	for jacknife in jacknifed_trials:
		est = recording.mask_epochs(jacknife)
		val = recording.select_epochs(jacknife)
		result = fit_model(est, val, model)
		result.append(result)

	plot_result(result)	
	publish_paper(result)

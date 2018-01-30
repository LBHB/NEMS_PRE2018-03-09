# Signals

`Signals` are the fundamental objects for storing timeseries data. The represent a value that changes in time, like the volume level of a speaker, the voltage across the membrane of a neuron, or the movement of an insect. They are objects designed to be useful for signal processing operations.

You can create `Signal` objects from a file, from a matrix, or from another `Signal`. 

An important note for beginners: once created, a `Signal` object cannot be changed -- it is /immutable/, just like tuples in Python. This is intentional and enables specific optimizations while also preventing entire classes of errors. But don't worry: we have made it easy to create new `Signal` objects based on other ones.


## Loading Signals from files

The majority of the time, you will be loading a `Signal` stored on disk. At a minimum, you will need a CSV file (which holds the tabular data, like a 2D matrix) and a JSON file, which stores metadata and documents what kind of data is in the CSV file.

Optionally, you will also often have an "epochs" file, which helps tag interesting events or periods of time in the timeseries for later analysis, but we will defer [the detailed documentation of epochs](epochs.md) until later.

Loading a `Signal` from a file is trivial:
```
from nems.signal import Signal
sig = Signal.load('/path/to/my/signalfiles'))
```

Creating your own CSV files is also pretty straightforward, but you need to understand the format. Read on if that interests you, or jump ahead if you would rather make it from a numpy array.

### Example Signal CSV File

For this example, make a new directory in the `signals/` directory called `testrec`because we are pretending we made a test recording. Inside that directory, make a file called `testrec_pupil.csv` and put the following inside the file:

```
2.0, 2.1
2.5, 2.5
2.3, 2.3
2.4, 2.5
2.4, 2.3
2.3, 2.4
```

In the CSV file, each row represents an instant in time, and each column is a "channel" of information of this signal. Channels can be anything you want -- they are just there to help you group several dimensions together.

We'll pretend the first channel is the diameter of a test subject's left pupil and the second channel is their right eye. There are only two channels and six time samples here, but in many experiments you will have tens of channels and thousands or millions of time samples.

### Example Signal JSON File

Continuing our example, let's make a JSON file that describes the contents of the CSV file containing our pupil data.

In the `signals/testrec/` directory, make another file called `testrec_pupil.json` and fill it with:

```
{"recording": "testrec", "name": "pupil", "chans": ["left_eye", "right_eye"], "fs": 0.1, "meta": {"Subject": "Don Quixote", "Age": 36}}
```

Here,

- `recording` is the name of the recording. We group collections of signals into "recordings", which is a name. 
- `name` is the name that you want to call this signal. 
- `fs` is the sampling rate in Hz. Generally it will be 10, 50, or even 00Hz, but for our test example, we assume that a measurement of the pupil diameter was only taken every 10 seconds, so `fs=0.1`.
- `chans` is the name of each channel (i.e. column in the CSV file), from left to right. 
- `meta` is extra information about the recording, such as the time of day it was taken, the experimenter, the subject, their age, or other relevant information. You may place anything you want here as long as it is a valid JSON data structure.

### Loading Example CSV + JSON

Assuming that your signal directory looks like this:

```
── testrec/
   ├── testrec.pupil.csv
   ├── testrec.pupil.json
```

You should now be able to load the pupil signal by making a file called `scripts/pupil_analysis.py` with the contents:

```
from nems.signal import Signal

# Note that we don't append the suffix .json or .csv
# because we are loading two files simultaneously
sig = Signal.load('../signals/testrec/testrec_pupil'))
```

That's it! You can start using your `Signal` now.

## Creating Signals from Other Signals

It's really common to make one signal from another signal.

TODO

## Creating Signals from Numpy Arrays

This is the least common and least recommended way of creating signals, but it definitely has its applications.

TODO

## Closing Thoughts on Signals

If you want to have a model that uses the data from 100 neurons, you can either have a single 100-channel Signal, or 100 one-channel signals. It's up to you.

Future work tickets:

- TODO: Rasterizing signals from an spike time list
- TODO: 


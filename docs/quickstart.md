# Quick Start

## What's an example script for analyzing data?

We have a good tutorial/template at: `scripts/demo_script.py`. 

You should be able to launch it with:

```
# Run this every time you open a terminal, or put in your .bashrc
export PYTHONPATH="$PYTHONPATH:/path/to/nems"

# Run the demo script
python3 scripts/demo_script.py
```


## I just want to analyze data! Where should I put my script?
   
If you are just hacking around, please put your code in `scripts` -- it's for any one-off analysis or snippet of code that isn't yet ready for reuse by other people.

The `tests/` directory is intended for unit tests that we can run automatically with `pytest`. 


## Where should I put my data?

If your data is in [Signal](signals.md) format, put it in the `signals/` directory. If it is not yet signal form, you may want to make a conversion script in `scripts/` that is able convert your data from your custom format into a Signal. 

Below is some pseudocode for a script that converts your data from a custom format and then save it as a Signal to make it easier for other people to use. 

```
from nems.signal import Signal

numpy_array = load_my_custom_data_format(...)
sig = Signal(matrix=numpy_array, 
             name='mysignal',
             recording='some-string', 
             fs=200  # Hz
             )
sig.save('../signals/my-new-signal')
```

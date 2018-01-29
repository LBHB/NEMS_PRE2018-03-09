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

## I want to analyze data! Where should I put my script?
   
If you are hacking around, please put your code in `scripts`. The `tests/` directory is intended for unit tests that we can run automatically with `pytest`. 


## Where should I put my data?

If your data is in Signal form, put it in the `signals/` directory. If it isn't in signal form yet, you can create one and then save it as a Signal if you want, to make it easier for other people to use:

```
from nems.signal import Signal

mat = load_my_matrix(...)
sig = Signal(matrix=mat, 
             name='mysignal',
             recording='some-string', 
             fs=200  # Hz   
             )
sig.save('/path/to-some/directory/')
```

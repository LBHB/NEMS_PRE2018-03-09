# NEMS #

NEMS is the Neural Encoding Model System. It is helpful for fitting a mathematical model to time series data, plotting the model predictions, and comparing the predictive accuracy of multiple models. We use it to study [computational models of how auditory stimuliare neurally encoded in mammalian brains](https://hearingbrain.org), but it will probably work with your timeseries data as well.


## Table of Contents ## 

1. [Quick Start](docs/quickstart.md)
2. Organizing your Data
   - [Signals](docs/signals.md)
   - [Recordings](docs/recordings.md)
3. Organizing your Models
   - [Modelspecs](docs/modelspecs.md)
4. Fitting your Models
   - [Fitters](docs/fitters.md)
5. Detailed Guides (TODO)
   - Creating your own modules
   - Comparing your models with others
   - Sharing modules, models, and data with others
6. Contributing to NEMS
   - [Project Goals](docs/goals.md)
   - [Design Discussion Issues](docs/discussions.md)
   - [Development History](docs/history.md)


## Work In Progress ##

Currently, these tasks are not yet complete:

0. Importing Baphy data as spiketimes
0. Inter-module data exchange occuring as Recording objects
1. Unit tests, in the `tests/` directory and run with `pytest`
2. Partial-stack optimizations during fitting, memoization
3. Ways of queueing fits and saving results
4. Ways of starting NEMS from the command line, and running a modelspec + data through several analyses
5. Theano integration
6. Contributor Guidelines

# README #

NEMS is the Neural Encoding Model System, a set of tools for fitting computational models for the neural encoding of sensory stimuli.  Written in Python, migrated from a Matlab tool designed to do something similar (NARF)

### Technical overview ###

General approach to creating and fitting a model

1.  Create an instance of the nems_stack
1.  Populate with a sequence of nems_modules
1.  Initialize the input to the first module with a nems_data_set
1.  The nems_data_set consists of one or more nems_data objects, which
correspond to stimulus/response data from a single experiment

Ongoing: expand this information in [NEMS Wiki](https://bitbucket.org/lbhb/nems/wiki/Home)

### Core components ###

* Model engine
    * Modules for each step of stimulus-response transformation
    * Data loader modules
    * Split/join state modules
    * Cost function/error modules
* Fitter - generic framework for updating model parameters based on cost function
    * Boosting/coordinate descent
    * Nested fitter (per module)
    * Jackknife fits (20 X 95% fit, 5% val)
* GUI
    * cellDB (other DB) interface
    * model inspector
    * results browser
    * population summary/comparison plots

### How do I get set up? ###

* Dependencies: numpy, scipy. Eventually pandas, sqlalchemy, some GUI support

### Who do I talk to? ###

* LBHB team
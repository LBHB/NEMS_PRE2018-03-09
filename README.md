# README #

NEMS is the Neural Encoding Model System, a set of tools for fitting computational models for the neural encoding of sensory stimuli.  Written in Python, migrated from Matlab tool (NARF)

### Technical overview ###

General approach.
1. Create an instance of the nems_stack
2. Populate with a sequence of nems_modules
3. Intialize the input to the first module with a nems_data_set
4. The nems_data_set consists of one or more nems_data objects, which
correspond to stimulus/response data from a single experiment

### Core components ###

* Model engine
* * modules simulating each step of transformation
* * data loader
* Fitter
* GUI

### How do I get set up? ###

* Summary of set up

### Who do I talk to? ###

* LBHB team
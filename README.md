# README #

NEMS is the Neural Encoding Model System, a set of tools for fitting computational models for the neural encoding of sensory stimuli.  Written in Python, migrated from a Matlab tool designed to do something similar (NARF)

### Technical overview ###

General approach to creating and fitting a model

1.  Create an instance of the FERReT (Field Encoding Response Regression Training) object. This should be give a "queue" of modules to fit,
    such as ('gammatone18ch','input_log','FIR','pupil_gain','tanhsig')
2.  Call the FERReT object attribute run_fit, with the desired fraction of the data to be reserved for validation and the number of 
    repetitions of fitting through the queue. Currently the vaildation data is just taken as a fraction of the stimuli from the end of the
    data set, but this will change to a jackknife fit later on. 
3.  Call the FERReT object attributes apply_to_val and apply_to_train to calculate the predicted responses to various stimuli using the fitted parameters.
4.  Various features of the data can be plotted, such as a raster of the raw data, the predicted response vs. the actual response to a stimulus (both for
    individual trials and averaged across all trials), and the stimulus intensity across each channel (working to create a sepctrogram)

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

* Dependencies: numpy, scipy, scipy.signal, scipy.io, math, matplotlib.pyplot, importlib, copy. Eventually pandas, sqlalchemy, some GUI support

### Who do I talk to? ###

* LBHB team
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
    * modNEM.py: Contains the FERReT object class. Currently has all the "logistics" aspects of model fitting, including cost function and error, as well as
      attributes that run a fit, apply the fitted parameters to data, and the actual fitting modules (though these may be modularized later).
    * function_pack: Package containing modules for each step of stimulus-response transformation
    * imports_pack: Package containing data loader modules
    * Plotting: Package containing modules for making plots. These should be functional both as stand-alone modules and as modules for the FERReT object.
        - raster_plot.py: generates raster plots of the reponse raster data. Highlights when stimulus was playing during the trial.
        - coeff_heatmap.py: generates a heatmap of module coefficients. This is particularly useful for visualizing FIR filters. 
        - comparison.py: generates plots that compare the predicted response to the actual response. There are two functions, one to show the response
          for individual trials and one to show the average response for all trials of a stimulus.
    * Split/join state modules (??)
    * Cost function/error modules (currently in the FERReT object class)
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

* Dependencies: numpy, scipy, scipy.signal, scipy.io, math, matplotlib.pyplot, importlib, copy, pandas, sqlalchemy, flask, mpld3, bokeh, pymysql, some GUI support

### Who do I talk to? ###

* LBHB team
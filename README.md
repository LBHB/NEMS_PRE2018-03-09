# README #

NEMS is the Neural Encoding Model System, a set of tools for fitting computational models for the neural encoding of sensory stimuli.  Written in Python, migrated from a Matlab tool designed to do something similar (NARF)

### Technical overview ###

General Overview:

1. NEMS models are described by keyword strings, e.g. 'fb18ch100_wc03_fir10_dexp_fit01_nested10'. These keywords tell the fitter how to load the data, which model functions to include in the model, and how to fit the model.
2. The NEMS models are essentially a modified stack, with each level of the stack being the output of a function applied to the previous element of the stack.
3. Models can be fit either on an individual computer or through the NEMS Analysis web app at http://neuralprediction.org:8000/ .


Ongoing: expand this information in [NEMS Wiki](https://bitbucket.org/lbhb/nems/wiki/Home)

### Core components ###

* Model engine:
    * nems_main.fit_single_model: the workhorse modelfitting function. Fits a single keyworded model to a single cell. Can do fits with either fixed validation sets or using cross-validation/jacknife fitting.
    * nems_stack: the actual data structure that composes the NEMS model. As the name suggests, the data is contained in a modified stack. Each entry in the stack is a dictionary containing several entries with relevant data. The nems_stack also has a modules attibute, which describes the module associated with each entry in the stack.
    * nems_module: an object class that describes the functions that operate on the stack. These functions ("modules") manipulate the cell data in various ways, from loading in the raw data to the stack object, to creating FIR filters and nonlinearities.
    * nems_keywords: keyword names for the NEMS models. These keywords are essentially helper functions, and append different modules to the model stack.
* Fitters - generic framework for updating model parameters based on cost function. Fitters are contained in their own separate object, nems_fitter, but are appended as a keyword like other model functions. They have their own object class nems_fitters. There are a few different fitters currently being used, although hopefully more will follow.
    * coordinate_descent: In-house boosting/coordinate descent algorithm
    * basic_min: framework for scipy.optimize.minimize that incorporates it into NEMS
    * anneal_min: simulated annealing fitter that utilizes scipy.optimize.basinhopping
    * fit_iteratively: fitter that fits each module individually using a specified fitter, such as basic_min
    * fit_by_type: fitter that fits each module individually using a fitter specified for each type of module, such as nonlinearity or weight channel
    * Jackknife fits (20 X 95% fit, 5% val) are taken care of at a higher level than individual fitters, and any of the fitters will work with the jacknife/cross-validation routine.
#Nested Crossvalidation - nested crossval (also called jacknife fitting) is implemented as keywords that are appended on the end of any modelname. They are contained in nems/nested.py, but are called by fit_single_model just like any other keyword. Must be at the end of the modelname.


### How do I get set up? ###

* Dependencies

    * Model fitting: numpy, scipy, scipy.signal, scipy.io, scipy.stats, scipy.special, matplotlib.pyplot, importlib, tensorflow, scikit-optimize
    * database-specific: pandas, sqlalchemy

* Demos

    * There are demo modules located in nems/modules/user_def/demo.py. These show how to make both simple and more advanced nems_modules, which are used to implement transformations to the data.
    * The basic_min nems_fitter is heavily annotated, and is the simplest implementation of a nems_fitter object. It should serve as a good jumping off point for writing other nems_fitter objects. It is located in nems/fitters/fitters.py.

### Who do I talk to? ###

* LBHB team

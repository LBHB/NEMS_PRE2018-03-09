# README #

## GOALS ## 

The goals of the "dev" branch are:

1. BAYESIAN PARAMETER SUPPORT. We want to estimate parameter distributions, not just a single value of a parameter. This lets us ditch nested crossvalidation, which is a huge complexity monster.
2. STANDALONE OPERATION. This branch should not depend on any databases and should run without an internet connection.
3. NO KEYWORDS. At least, not yet. We're going to try to use JSONs instead so we can control parameters better.
4. MINIMALISM. We are going to try to keep this branch as succinct as possible for the moment;

For the moment, we are ignoring queuing models on the cluster, keywords, web interfaces, and most of the functionality in the master branch. We will migrate this over later.


## DESIGN DISCUSSION POINTS ##

(From a slack discussion:)

STACK. The stack is now renamed the "Model" class.

DB ACCESS. Direct mysql DB access from NEMS is forbidden; instead, assume you have the files on disk (they can be fetched via HTTP, jerbs, DAT, or whatever)

KEYWORDS. Keywords are discouraged for the moment, because they were doing too many things at once: A) Appending of a module to a model; B) Initializing module parameters, which is better done with a JSON that describes prior beliefs about the values; C) Fitting module parameters. We may bring back keywords later, but for the moment, we'd like to move towards using JSONs to create models, so that we can fully specify parameter priors and rely less on automatic "initial fits" to get priors.

MODULES MAY NOT INTROSPECT THE STACK. Modules are now forbidden from knowing anything about the rest of the stack. If you can think of a good reason why they absolutely need to see the rest of the stack, bring it up ASAP; otherwise we are going to plan to disallow this. Yes, this implies that STRF plots (which need multiple models) should be done by a function that takes in a model as an argument, not by a module method.

SIGNAL DIMENSIONS. I'd like to propose a simplification. In my opinion, the simplest and most general case for all data is for Signal objects to always be 2D: channel x time_sample, like a CSV file with each row being a moment in time. Trial-by-trial groupings shouldn't matter for the evaluation of a module (I hope?), and not supporting 4D arrays gets rid of nasty problems that occur at the boundaries of a convolved signal when it is folded by trial.  Basically, then, if you want to have a model that uses the data from 100 neurons, you can either have a 100 channel Signal, or 100 one-channel signals. It's up to you.

SPLITTING DATA. I don't think data splitting should be part of the Fitter class -- data selection is a judgement call by the programmer before the fitting begins. You may want to use 3 data files as estimation data, and then use one for validation data. Or, you may want to use 60% of a single file as estimation data, and then 40% as validation. It really varies, and depends on the analysis. Also, some fitters accidentally have used the entire data set, not just the estimation data set; we should try to avoid this class of problems by keeping splitting out of the fitter.

FITTERS.  This is the hardest part in my experience and needs the most thought. Just to muddy the waters a bit, here are some things that came up in the past:
  - People inadvertently cheating by using the whole data set for fitting, instead of just the estimation data set.
  - Fitting roughly at first with one algorithm, then fitting with another to get a final fit (it's stupid, but it works better)
  - Iterating over all modules, fitting only parameters from one module at a time (it's stupid, but it works often)
  - Fitting subsets of the parameters (to avoid n^2 performance penalties with some fitters)
  - Plugging in different cost functions for the same fitter (L1 vs L2 vs Log Likelihood)
  - Using one cost function for fitting, but multiple cost functions (metrics) for evaluating the final performance
  - Trying different termination conditions. Usually a predicate function that returns true when you should stop fitting. Reasons to stop fitting include a certain number of model evaluations, gradient step size, average change in error, too many NaN predictions, or elapsed time.
  - A helpful performance optimization in NARF was to avoid recomputing the entire stack; only recompute modules whose parameters were changed or had previous modules with changed parameters.


## WORK PLAN ##

### Ivar ###
    - Ivar will start a "dev" branch, and put the Signal class object in it
    - Ivar will work on an API to get Baphy data as Signal objects

### Brad ###
    - Brad is going to add his Bayesian template to the dev branch, then use PyMC3 and Theano to the modules to get some bayesian fitters working (maybe a trivially simple model first?)
    - Brad will flesh out modules, and their conversion to and from JSONs
    - Brad will come up with some results that are convincing for SVD! ;-)

### Jake ### 
    - Jake and Ivar will write a generic fitter class together, and discuss how to remove cross validation from all modules
    - Jake will take notes and test out various fitters, taking metrics on runtime and MSE (Adagrad, Adam, RMSProp, etc)
    - Jake will reconnect the plotting routines and hooking it up to the web interface again
 
### Later ###
    - After we have a "toy" version of Bayesian NEMS working, we will talk about how to migrate the modules in master branch over to "dev". At this point we will remove all CV details from modules and generally try to keep "dev" minimalist
    - Unit tests
    - Command line arguments
    - Partial-stack optimizations for fitting
    - Ways of queuing fits and saving results
    - A fitter that uses a custom sub-fitter for each module of a model
    - Theano module
    - Supporting "drop-out" of parameters/neurons in a multi-neuron fit
    - Supporting Cross Validation


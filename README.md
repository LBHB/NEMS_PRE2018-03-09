# README #

## GOALS ## 

The goals of the "dev" branch are:

1. BAYESIAN PARAMETER SUPPORT. We want to estimate parameter distributions, not just a single value of a parameter. This lets us ditch nested crossvalidation, which is a huge complexity monster.
2. STANDALONE OPERATION. This branch should not depend on any databases and should run without an internet connection.
3. MINIMALISM. We are going to try to keep this branch as succinct as possible for the moment;

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
    - Ivar will work on an API to get Baphy data as Signal objects

### Brad ###
    - Brad is working on the RDT models and can't work on NEMS for a while.

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


# Stateless Module Demo

# The more I think about it, the more I wonder if our real goal should just
# be to exorcise all state from a module, turning modules into "pure functions"
# (a.k.a. functions without side effects). Not that we shouldn't use classes,
# but that we should keep focused on stateless functions when possible.
#
# Managing modules can be complicated precisely because they contain mutable
# state. Given that state is usually easier when it is all in once place,
# maybe packing the entire model into a single data structure isn't such a
# crazy idea.   
#
# The following shows a little demo of how that might look in general,
# and for three cases that are not supported by the current version of NEMS:
#
#   1. "Per-module fitters", when each module uses a different sub-fitter
#      and all modules are iterated through.
#
#   2. "Parameter fitting dropout", when some parameters are randomly NOT
#      considered for optimization during the fitting process
#
#   3. "Module dropout", when the transformation of random modules are
#      temporarily omitted during the fitting process
#
# The latter two techniques have been used extensively in deep learning
# and allegedly make the fitting process more robust. And per-module
# fitters could be a cleaner way of putting all the fitting code in a single
# place rather than spreading it out and putting a little in each module,
# which then requires that other fitters remove manually whenever they
# do not want to do an "initial-fit".

##############################################################################

Are you proposing that a single modelspec will contain all of that information? I would suggest breaking it out a little and have separate data structures. One describes the model, one is a Recording containing the data, one is your fit function (which you have carefully composed from the pieces described in fitter_dataflow.svg, perhaps using a fitterspec to help quickly compose the pieces) and phi (the value of all model cofficients). 

Q: Why is Phi part of the modelspec? I have to clone the modelspec each time I want to run a different fit on the model!

A: I'm conceptually thinking about modelspecs as being something that defines the entire input-output "black box" model; yes, the parameters (phi) are a special case in many cases, but they still fall within the black box and can't be logically separated from it without having to lug around the knowledge that this phi goes with that black box, and this other phi goes with that other black box. I'm willing to pay the very slight extra memory use because I think we can optimize it away in other ways.

-----

Let me clarify how it works in my code

Every  Module implements a get_priors method that takes one argument (the data being fit). The module may use this data to help come up with reasonable information about the fit bounds. For each parameter, a distribution is returned. This distribution defines the min/max and range of possible values.

So, for a value that can take on any positive values, you'd use a Gamma distribution. The mean of the gamma distribution (E[x] = alpha/beta) will be set to what you think is a reasonable value for that parameter. The fitter can then choose to set the initial value for the parameter to E[x] or draw a random sample from the distribution.

For a value that can take on any value, you'd use a Normal centered at what a reasonable expected starting point for the value is.

For a value that must fall between 0 and 1 you can choose either a Uniform or Beta distribution.

For parameters that are multi-dimensional (e.g. FIR coefficients and mu in weight channels), the Priors can be multidimensional as well. So, for weight channels you can specify that the first channel is a Beta distribution such that the channel most likely falls at the lower end of the frequency axis and the second channel at the upper end of the frequency axis.


------

modules/sum.py. Should this be renamed "sum_channels.py"? We might have a 'sum_signals.py" module at some point. Also: should this summing implementation be put in the "signals" object, which we then call from this file, in order that we don't accidentally have two similar-but-not-identical versions of the same code? (I guess the answer to this depends on whether signals are passed between modules or not, as the same problem comes up with a "normalization" module and the Signal.normalize() methods)

----

Fitter Input Argument Specs. I think I may be arguing with my past self here, but I am wondering if we can remove the need to pass the "model" object to our fitting algorithms? I would ideally just prefer to have fitters accept a cost function, instead of having any knowledge about the model structure. I feel like any optimizations (evaluating part of the stack, per-module fitters) could still be accomplished with carefully structured functional composition.

---

Inter-module Data Exchange Format. Now that we have Signal objects, have we decided the data type once and for all? Numpy arrays? Or Signal/Recording objects? The former is probably more efficient, the latter is (debatably) more convenient for interoperability. Since the signal object was not available before, I can see that Brad assumed numpy arrays would be exchanged -- is that necessary for Theano to work?

---

Lazy Devil's Advocate. To rethink a design decision, is it really worth wrapping all of the scipy.stats distributions with nems.distributions.* instead of instead of using them directly? What specific advantages do we get from this?

---

1) I added a docs folder and moved some of the planning documentation to docs/planning. I also added an IPython notebook that explains how I envision distributions working in the NEMS ecosystem. Bitbucket doesn't let you view the formatted notebook (Github does), so you can view it using this link: https://nbviewer.jupyter.org/urls/bitbucket.org/lbhb/nems/raw/b962df365a79f2c68dabd7d575e6fca05ea474ec/docs/distributions.ipynb

2) I have functional versions of the modules, fitters and model portions of the system right now. To see how we can implement it using a bayes approach vs scipy, compare nems.fitters.scipy and nems.fitters.pymc3. The bayes approach is a very abstract system and requires quite a bit of knowledge re how PyMC3 (the bayes fitting package) works, so I haven't documented it in depth. Basically PyMC3 uses symbolic computation to build a symbolic model, then evaluates it once it's built.

3) Stephen's very concerned about "mini-fits", so the iterative_fit function in the nems.fitters.scipy should hopefully alleviate his concerns.

4) I've made the fitting routines functions (i.e., functional approach) rather than objects. It just seems to make more sense for these basic fits. There's no reason why some fitters can't be objects (e.g., if we are building a complex fitter with sub-fitters for each module and we need a central object to track the state).

5) I also tried to document as much as I could so that you can hopefully follow my logic. Let me know if you have any questions or if anything is unclear.

6) I haven't fully updated the code to work with some of the changes you made (e.g., I used some hacks since the Signal and Recording objects weren't production ready). I'm going to spend the next few hours looking into these changes and try to sync up everything so it works with the new data-loading system.

---
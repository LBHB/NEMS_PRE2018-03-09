# Design Discussions # 

The following are a list of design discussion points that were brought up in our Slack conversations and may help inform why the design ended up the way that they did. It is probably not useful to non-developers. 

## DB ACCESS ##

Direct mysql DB access from NEMS is forbidden; instead, assume you have the files on disk (they can be fetched via HTTP, jerbs, DAT, or whatever)

## MODELSPECS ## 

The more I think about it, the more I wonder if our real goal should just be to exorcise all state from a module, turning modules into "pure functions"
(a.k.a. functions without side effects). Not that we shouldn't use classes,
but that we should keep focused on stateless functions when possible.

Managing modules can be complicated precisely because they contain mutable
state. Given that state is usually easier when it is all in once place,
maybe packing the entire model into a single data structure isn't such a
crazy idea.

The following shows a little demo of how that might look in general,
and for three cases that are not supported by the current version of NEMS:
  1. "Per-module fitters", when each module uses a different sub-fitter
     and all modules are iterated through.
  2. "Parameter fitting dropout", when some parameters are randomly NOT
     considered for optimization during the fitting process
  3. "Module dropout", when the transformation of random modules are
     temporarily omitted during the fitting process

The latter two techniques have been used extensively in deep learning
and allegedly make the fitting process more robust. And per-module
fitters could be a cleaner way of putting all the fitting code in a single
place rather than spreading it out and putting a little in each module,
which then requires that other fitters remove manually whenever they
do not want to do an "initial-fit".

## MODELSPECS AND PHI ## 

Q: Why is Phi part of the modelspec? I have to clone the modelspec each time I want to run a different fit on the model!

A: I'm conceptually thinking about modelspecs as being something that defines the entire input-output "black box" model; yes, the parameters (phi) are a special case in many cases, but they still fall within the black box and can't be logically separated from it without having to lug around the knowledge that this phi goes with that black box, and this other phi goes with that other black box. I'm willing to pay the very slight extra memory use because I think we can optimize it away in other ways.

## KEYWORDS ##

Our previous implementations of "keywords" were functions that could do anything. Now they are discouraged, because they were doing too many things at once: A) Appending of a module to a model; B) Initializing module parameters, which is better done with a JSON that describes prior beliefs about the values; C) Fitting module parameters. Our new strategy should be to use keywords as abbreviations that help create a modelspec. 


## MODULES MAY NOT INTROSPECT THE STACK ##

Modules are now forbidden from knowing anything about the rest of the stack. If you can think of a good reason why they absolutely need to see the rest of the stack, bring it up ASAP; otherwise we are going to plan to disallow this. Yes, this implies that STRF plots (which need multiple models) should be done by a function that takes in a model as an argument, not by a module method.


## SIGNAL DIMENSIONS ##

I'd like to propose a simplification. In my opinion, the simplest and most general case for all data is for Signal objects to always be 2D: channel x time_sample, like a CSV file with each row being a moment in time. Trial-by-trial groupings shouldn't matter for the evaluation of a module (I hope?), and not supporting 4D arrays gets rid of nasty problems that occur at the boundaries of a convolved signal when it is folded by trial.  Basically, then, if you want to have a model that uses the data from 100 neurons, you can either have a 100 channel Signal, or 100 one-channel signals. It's up to you.


## SPLITTING DATA ## 

I don't think data splitting should be part of the Fitter class -- data selection is a judgement call by the programmer before the fitting begins. You may want to use 3 data files as estimation data, and then use one for validation data. Or, you may want to use 60% of a single file as estimation data, and then 40% as validation. It really varies, and depends on the analysis. Also, some fitters accidentally have used the entire data set, not just the estimation data set; we should try to avoid this class of problems by keeping splitting out of the fitter.


## FITTERS ## 

 This is the hardest part in my experience and needs the most thought. Just to muddy the waters a bit, here are some things that came up in the past:
  - People inadvertently cheating by using the whole data set for fitting, instead of just the estimation data set.
  - Fitting roughly at first with one algorithm, then fitting with another to get a final fit (it's stupid, but it works better)
  - Iterating over all modules, fitting only parameters from one module at a time (it's stupid, but it works often)
  - Fitting subsets of the parameters (to avoid n^2 performance penalties with some fitters)
  - Plugging in different cost functions for the same fitter (L1 vs L2 vs Log Likelihood)
  - Using one cost function for fitting, but multiple cost functions (metrics) for evaluating the final performance
  - Trying different termination conditions. Usually a predicate function that returns true when you should stop fitting. Reasons to stop fitting include a certain number of model evaluations, gradient step size, average change in error, too many NaN predictions, or elapsed time.
  - A helpful performance optimization in NARF was to avoid recomputing the entire stack; only recompute modules whose parameters were changed or had previous modules with changed parameters.

## PRIORS AS SETTING BOUNDS ##

Every Module implements a get_priors method that takes one argument (the data being fit). The module may use this data to help come up with reasonable information about the fit bounds. For each parameter, a distribution is returned. This distribution defines the min/max and range of possible values.
  - So, for a value that can take on any positive values, you'd use a Gamma distribution. The mean of the gamma distribution (E[x] = alpha/beta) will be set to what you think is a reasonable value for that parameter. The fitter can then choose to set the initial value for the parameter to E[x] or draw a random sample from the distribution.
  - For a value that can take on any value, you'd use a Normal centered at what a reasonable expected starting point for the value is.
  - For a value that must fall between 0 and 1 you can choose either a Uniform or Beta distribution.
  - For parameters that are multi-dimensional (e.g. FIR coefficients and mu in weight channels), the Priors can be multidimensional as well. So, for weight channels you can specify that the first channel is a Beta distribution such that the channel most likely falls at the lower end of the frequency axis and the second channel at the upper end of the frequency axis.

## SUM_CHANNELS ##

Should this be renamed "sum_channels.py"? We might have a 'sum_signals.py" module at some point. Also: should this summing implementation be put in the "signals" object, which we then call from this file, in order that we don't accidentally have two similar-but-not-identical versions of the same code? (I guess the answer to this depends on whether signals are passed between modules or not, as the same problem comes up with a "normalization" module and the Signal.normalize() methods)


## Fitter Input Argument Specs ## 

I think I may be arguing with my past self here, but I am wondering if we can remove the need to pass the "model" object to our fitting algorithms? I would ideally just prefer to have fitters accept a cost function, instead of having any knowledge about the model structure. I feel like any optimizations (evaluating part of the stack, per-module fitters) could still be accomplished with carefully structured functional composition.

## Inter-module Data Exchange Format ##

Now that we have Signal objects, have we decided the data type once and for all? Numpy arrays? Or Signal/Recording objects? The former is probably more efficient, the latter is (debatably) more convenient for interoperability. Since the signal object was not available before, I can see that Brad assumed numpy arrays would be exchanged -- is that necessary for Theano to work?

## Lazy Devil's Advocate ##

Q: To rethink a design decision, is it really worth wrapping all of the scipy.stats distributions with nems.distributions.* instead of instead of using them directly? What specific advantages do we get from this? 

A: It's easier for us to control the behavior if we wrap the distributions. For example, look at nems.distributions.distributions:Distribution.sample. It's not just a simple mapping to the underlying scipy.stats distribution.

## SCIPY ## 

I have functional versions of the modules, fitters and model portions of the system right now. To see how we can implement it using a bayes approach vs scipy, compare nems.fitters.scipy and nems.fitters.pymc3. The bayes approach is a very abstract system and requires quite a bit of knowledge re how PyMC3 (the bayes fitting package) works, so I haven't documented it in depth. Basically PyMC3 uses symbolic computation to build a symbolic model, then evaluates it once it's built.

## ITERATIVE FITS ##

Stephen's very concerned about "mini-fits", so the iterative_fit function in the nems.fitters.scipy should hopefully alleviate his concerns.


## FUNCTIONAL FITTERS ##

I've made the fitting routines functions (i.e., functional approach) rather than objects. It just seems to make more sense for these basic fits. There's no reason why some fitters can't be objects (e.g., if we are building a complex fitter with sub-fitters for each module and we need a central object to track the state).


## ON THE NAMES OF FUNCTIONS ## 

To help with clarity, we will define the following words mathematically:

```
 |-----------+----------------------------------------------------------|
 | Name      | Function Signature and Description                       |
 |-----------+----------------------------------------------------------|
 | EVALUATOR | f(mspec, data) -> pred                                   |
 |           | Makes a prediction based on the model and data.          |
 |-----------+----------------------------------------------------------|
 | METRIC    | f(pred) -> error                                         |
 |           | Evaluates the accuracy of the prediction.                |
 |-----------+----------------------------------------------------------|
 | FITTER    | f(mspec, cost_fn) -> mspec                               |
 |           | Tests various points and finds a better modelspec.       |
 |-----------+----------------------------------------------------------|
 | COST_FN   | f(mspec) -> error                                        |
 |           | A function that gives the cost (error) of each mspec.    |
 |           | Often uses curried EST dataset, METRIC() and EVALUATOR() |
 |-----------+----------------------------------------------------------|

where:
   data       is a dict of signals, like {'stim': ..., 'resp': ..., ...}
   pred       is a dict of signals, just like 'data' but containing 'pred'
   mspec      is the modelspec data type, as was defined above
   error      is a (scalar) measurement of the error between pred and resp
```

## WHERE SHOULD THE DATASPEC BE RECORDED? ## 

TODO: Open question: even though it is only a few lines, how and where should this information be recorded? The data set that a model was trained on is relevant information that should be serialized and recorded somewhere.

```
 save_to_disk('/some/path/model1_dataspec.json',
              json.dumps({'URL': URL, 'est_val_split_frac': 0.8}))
```

TODO: This annotation should be done automatically when split_at_time is called?

## Splitting, Jackknifing, and Epochs

@jacob In reply to your excellent question about what we should do for jackknifed_by_epochs and splitting based on epochs, and what data formats those should return, I think I made a mistake in asking for regex matching as part of the core functionality, and I'd like to walk that back a bit.

On the dev branch, I basically just removed the "regex" matching from split_at_epoch things, and things just worked fine. I didn't fix jackknife_by_epochs yet, and I'm not entirely sure what the right way to do that is, and I'm open to ideas. My current hunch is to make it more like jackknife_by_time, and I'm guessing that rounding to the nearest occurence of by_epochs would be the way to do it (and warn if the rounding is off results in partitions that, say, differ more than some critical amount). But I'm open to ideas. 
 
Now, I still think regex functionality is cool, but after talking with SVD, I'm thinking we should do that in a single function, like `signal.match_epochs('regex')` which will give us a list of all matching epochnames that we can then iterate through. 

Something like:

```
TORCs = signal.match_epochs('^TORC.*')

for torc in TORCs:
    my3dmatrix = signal.fold_by(torc)
    mean_for_this_torc = numpy.mean(my3dmatrix, axis=0)
    plot(mean_for_this_torc)
```

Mostly, I just wanted to avoid 4D matrices since they make my head hurt when they get ragged. 



 and Let's make trial_epochs_from_reps return 


Honestly, I'm not sure. 




 the new functionality of epoch s


## What about split
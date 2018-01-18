# ----------------------------------------------------------------------------
# DEFINE A MODELSPEC

# GOAL: Uniquely define the structure of the model you wish to fit or use to
# make a prediction about your data. A 'modelspec' is a datastructure that
# defines the entire model and is easily saved/loaded to disk.
# It is essentially a serialization of a Model object and may be defined using:

# Method #1: create from "shorthand/default" keyword string
# modelspec = keywords_to_modelspec('fir30_dexp')

# Method #2: load a JSON from disk
# modelspec = json.loads('modelspecs/model1.json')

# Method #3: Load it from a jerb (TODO)
# modelspec = ...

# Method #4: specify it manually

from nems.modules.spec.templates import plot_pred_spec

modelspec = [
    {
        # Name
        'name': 'Gaussian weighting of channels'
        'id': 'wcg02'
        'description': 'Some description goes here'

        # Base import path that can be used to find the various functions
        # defined next (makes it easier to refactor the namespace if we want to
        # move the weight channels module inside a module called "filters".
        'fn_base': 'nems.modules.weight_channels.gaussian',

        # Due to fn_base, the functions below will be found in the module
        # nems.modules.weight_channels'. All functions have the following
        # signature:
        #
        #     def function(recording, input_name, output_name, n_outputs)
        #        pass

        # This function is mandatory. It takes a recording object (along with
        # the keyword arguments specified below) and returns a new recording
        # object.
        'fn': 'numeric_evaluation',

        # Optional. Additional kwargs to be passed into numeric_evaluation.
        # Since we can't do currying in a JSON-based modelspec, this is the
        # closest we can get. When building the model evaluation function, we
        # can curry these into the function.
        'fn_kwargs': {},

        # This function is optional and is used if we are using Theano or
        # TensorFlow to generate a graph. Note that sometimes the regular eval
        # function (`fn`) is satisfactory so we can just point to that instead.
        'fn_symbolic': 'symbolic_evaluation',

        # Optional kwargs to be curried into fn_symbolic
        'fn_symbolic_kwargs': {},

        # Function that returns a dictionary of parameters and the coefficients
        # for each parameter. This function can also return the initial values
        # for the parameters.
        'fn_coefficients': 'get_coefficients'

        # Optional kwargs to be curried into fn_coefficients
        'fn_coefficients_kwargs': {'init_method': 'random'},

        # This is optional and specifies the bounds for each coefficient. If not
        # provided, we will set the limits to [-Inf, +Inf]. Note that fn_limits
        # can optionally call fn_priors and use the priors to generate the
        # bounds.
        'fn_bounds': 'get_bounds_strict',

        # Optional and specifies priors that are used in Bayesian regression.
        'fn_priors': 'get_priors',

        # Optional and provides plotting functions that can be used with this
        # module.
        'fn_plot': [
            {
                'fn': 'plot_coefficients',
                'description': 'Plots coefficients'
                'label': 'Plot coefficients'
            },

            # Since plot_prediction is a generic function that can be used after
            # the output of almost any module, we can wrap it up into a spec
            # that can be reused easily (note that when exported to JSON, it
            # will get expanded to the full-blown spec).
            plot_pred_spec,
        ]

        # Generic set of kwargs that will be provided to each function (this is
        # where we can modify some of the key options).
        'fn_kwargs': {
            'input_name': 'pred',
            'output_name': 'pred',
            'n_outputs': 2,
        },
    },
    {
        'name': 'FIR filter'
        'id': 'fir15'
        'description': 'Some description goes here'

        'fn_base': 'nems.modules.fir',
        'fn': 'numeric_evaluation',
        'fn_symbolic': 'symbolic_evaluation',
        'fn_coefficients': 'get_coefficients'
        'fn_coefficients_kwargs': {'init_method': 'zero'},
        'fn_priors': 'get_priors',
        'fn_plot': 'plot_coefficients',
        'fn_kwargs': {
            'input_name': 'pred',
            'output_name': 'pred',
            'n_taps': 15,
        },
    } ]

# To facilitate composition of a modelspec programatically, we can provide
# defaults. For example:

from nems.modules.weight_channels.gaussian import template as wc_template

wc_spec = wc_template.copy()
wc_spec['fn_kwargs']['n_outputs'] = 4

modelspec = [
    wc_spec,
    ...
]


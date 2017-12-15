#!/usr/bin/env python3

# Minimalist code for fitting models stored in CSV/JSON files

import os
import nems.Signal as sig
import nems.nobaphy_hackery as hack


# You may load your Signals in any way that you wish.
# 1. From a local file in the signals/ dir (perhaps placed there by jerbs)
# 2. Requesting a file over HTTP (so that credentials)
# 3. Using a DAT project live data stream.

# For now, we'll do #1: load all JSON/CSV pairs in the "signals" directory
# This is just a convention that simplifies composing jerbs
signals_dir = os.path.join(os.getcwd(), "signals")
signals = sig.load_signals_in_dir(signals_dir)

signals_found = [s.name for s in signals]
print("Signals found: ", signals_found)


# Now define a modelname (excluding the loader, which is no longer
# needed because we loaded the files manually)
modelname = 'wcg02_fir10_fit01'
stack = hack.build_stack_from_signals_and_keywords(signals, modelname)

# TODO: In the future, we should provide several ways of constructing models:
#   1. Creating stacks via keywords as we do now (but without hackery)
#   2. By loading a JSON (so that you can start with a saved model)
#   3. By instantiating a Model object (that may not be stack-based)

# TODO: In the future, we could now verify that the stack's required
# input signals are actually present in signals_found; this type of contract
# could help detect errors


# TODO: In the future, we should
print("Fitting...")
phi = stack.fitter.fit_to_phi()

print("Done with fit.")

# TODO: preview_file = stack.quick_plot_save(mode="png")

# TODO: serialize a model (i.e. save each module's name and parameters
# as a JSON, then combine all of the modules into a nested JSON containing
# the entire model)

# TODO: Save the JSON to a default directory

# TODO: Generate a JERB containing the serialized model and publish it

# TODO: Generate a JERB containing the preview image file (maybe?)

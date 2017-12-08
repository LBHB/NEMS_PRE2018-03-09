#!/usr/bin/env python3

# A fake, minimalist fit_single_model that loads signal files instead
# of requesting them from Baphy

import os
import nems.Signal as sig
import nems.hackery as hack


if __name__ == '__main__':

    # You may load your Signals in any way that you wish
    # 1. From a local file (perhaps via a Jerb)
    # 2. Loading directly over HTTP, S3, etc
    # 3. Some database query

    # For now, just put them in a single directory:
    signals = sig.load_signals_in_dir(os.getcwd())
    modelname = 'wcg02_fir10_pupwgtctl_fit01'

    stack = hack.build_stack_from_signals_and_keywords(signals, modelname)

    print("Fitting...")
    phi = stack.fitter.fit_to_phi()

    print("Done with fit.")

    stack.meta['n_parms'] = len(phi)

    # TODO: preview_file = stack.quick_plot_save(mode="png")

    # TODO: serialize a model (i.e. as a JSON), and save it

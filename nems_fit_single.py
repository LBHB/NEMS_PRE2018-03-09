#!/usr/bin/env python3

# This script runs nems_main.fit_single_model from the command line

import nems.keyword as nk
import nems.modules as nm
import nems.stack as ns
import sys


# A dataset is simply a set of Signals, and needs no wrapper I think

# A Stack is a sequence of modules, and has a wrapper object that:
#   Returns parameter sets
#   Evaluates given an input Signal, and spits out an output Signal

# A ModelConstructor is a sequence of functions that initializes, creates,
#   and fits the model to the dataset


def construct_model(keywordstring):
    """ Constructs a model from a sequence of keyword functions. """
    stack = ns.nems_stack(modelname=keywordstring)

    stack.meta['signalfiles'] = signalfiles
    stack.valmode = False
    stack.keywords = modelname.split("_")
    stack.keyfuns = nk.keyfuns
    for k in stack.keywords:
        stack.keyfuns[k](stack)
    stack.valmode = True
    stack.evaluate(1)
    stack.append(nm.metrics.correlation)

    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}"
          .format(stack.meta['mse_est'],
                  stack.meta['mse_val'],
                  stack.meta['r_est'],
                  stack.meta['r_val']))

    valdata = [i for i, d in enumerate(stack.data[-1]) if not d['est']]
    stack.plot_dataidx = valdata[0] if valdata else 0
    phi = stack.fitter.fit_to_phi()
    stack.meta['n_parms'] = len(phi)
    return(stack)


   python3 nems_fit_single.py gus027b-a1 293 \
   parm50_wcg02_fir10_pupwgtctl_fit01_nested5

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('syntax: nems_fit_single modelname signal1 [signal2...]')
        exit(-1)

    modelname = sys.argv[1]
    signalfiles = sys.argv[2:]

    print("Creating Model...")
    m = nems.Model(

    print("Starting fit.")
    stack = fit_model(modelname, signalfiles)
    print("Done with fit.")

    # TODO: Decide to plot or not
    # autoplot = True
    # if autoplot:
    #     stack.quick_plot()

    # TODO: Save result
    # filename = nems.utilities.io.get_file_name(cellid, batch, modelname)
    # nems.utilities.io.save_model(stack, filename)

    # Edit: added code to save preview image. -Jacob 7/6/2017
    # preview_file = stack.quick_plot_save(mode="png")
    # print("Preview saved to: {0}".format(preview_file))

    #r_id = nd.save_results(stack, preview_file, queueid=queueid)
    # print("Fit results saved to NarfResults, id={0}".format(r_id))

    # TODO: make new jerb from result?


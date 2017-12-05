import nems.keyword as nk
import nems.modules as nm
import nems.stack as ns


def build_stack_from_signals_and_keywords(signals, modelname):
    """ Docs: TODO """
    stack = ns.nems_stack()
    stack.append(nm.loaders.load_signals, signals=signals)
    stack.append(nm.est_val.crossval, valfrac=0.2)

    stack.meta['batch'] = '9999'
    stack.meta['cellid'] = 'dummy-cellid'
    stack.meta['modelname'] = modelname

    stack.valmode = False
    stack.keywords = modelname.split("_")
    stack.keyfuns = nk.keyfuns  # pre-indexed set of keyword functions

    # evaluate the stack of keywords
    if 'nested' in stack.keywords[-1]:
        # special case for nested keywords. Stick with this design?
        print('Using nested cross-validation, fitting will take longer!')
        k = stack.keywords[-1]
        stack.keyfuns[k](stack)
    else:
        print('Using standard est/val conditions')
        for k in stack.keywords:
            stack.keyfuns[k](stack)

    # measure performance on both estimation and validation data
    stack.valmode = True
    stack.evaluate(1)

    stack.append(nm.metrics.correlation)

    print("mse_est={0}, mse_val={1}, r_est={2}, r_val={3}"
          .format(stack.meta['mse_est'],
                  stack.meta['mse_val'],
                  stack.meta['r_est'],
                  stack.meta['r_val']))

    valdata = [i for i, d in enumerate(stack.data[-1]) if not d['est']]
    if valdata:
        stack.plot_dataidx = valdata[0]
    else:
        stack.plot_dataidx = 0

    return stack

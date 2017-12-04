#!/usr/bin/env python3

# A fake, minimalist fit_single_model that loads signal files instead
# of requesting them from Baphy

import nems.main as nems

if __name__ == '__main__':

    cellid = 'gus027b13'
    batch = '666'
    modelname = 'parm50_wcg02_fir10_pupwgtctl_fit01'

    print("Running fit_single_model({0},{1},{2})".format(cellid,
                                                         batch,
                                                         modelname))
    stack = nems.fit_single_model(cellid, batch, modelname, autoplot=False)
    print("Done with fit.")

    preview_file = stack.quick_plot_save(mode="png")
    print("Preview saved to: {0}".format(preview_file))

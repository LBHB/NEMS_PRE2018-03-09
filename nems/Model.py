import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nems.utilities.utils import find_modules
import numpy as np
import os
import io
import copy


class Model:
    """ A Model is essentially just a list of nems_module objects. Models
    accept input Signal objects and return a new output Signal object, (i.e.
    a prediction). Models also have instance methods to help use the relevant
    parameters buried in their internal state.

    Build a model using:
      Model([module1, module2, ...], KWARGS)

    where KWARGS may be:
      modelname  The name of this model. Default is to use module names
                 concatenated together with underscores.
      fitter     The fitting algorithm to use. Default is <TODO>
      output_signal_goal   The signal name to try to predict; the 'goal'
      output_signal_pred   The signal name created by the model prediction

    Instance Methods:
      .get_mask()       TODO
      .set_mask()       TODO: Sets the modules whose parameters are to be fit?
      .get_params()     Returns a list of all module parameters (phi).
      .set_params()     Sets the parameters (phi).
      .fit(TODO)        Fits the model to the data
      .plot_summary()   Returns a summary plot of the entire model.
      .modules()        Returns the ordered list of the modules.
    """
    def __init__(self,
                 modules=[],
                 modelname=None,
                 fitter=None,
                 meta=None,
                 output_signal_goal='resp',
                 output_signal_pred='pred'):
        self.modules = modules
        if not modelname:
            guessed_modelname = [m.name for m in modules].join('_')
        self.modelname = modelname if modelname else guessed_modelname
        self.fitter = fitter
        self.meta = meta if meta else {}

    def evaluate(self, start_idx=0, end_idx=None):
        """ Evaluate modules in model, starting at module # start_idx,
        and continuing to the last module or end_idx, whichever is sooner. """
        end_idx = end_idx if end_idx else len(self.modules)

        ms = self.modules
        # TODO: Can we avoid searching the stack somehow?
        xval_idxs = find_modules(ms, 'est_val.crossval')
        xval_idx = xval_idxs[0] if xval_idxs else None
        # TODO: Can we avoid searching for metrics somehow?
        mse_idxs = find_modules(ms, 'metrics.mean_square_error')
        mse_idx = mse_idxs[0] if mse_idxs else None

        for i in range(start_idx, end_idx):
            self.modules[i].evaluate()

    def append(self, mod=None, evaluate=True):
        """ Creates an instance of a module and appends it to the stack.
        Default is to evaluates module after doing so. """
        if mod is None:
            raise ValueError('stack.append: module not specified')

        self.modules.append(mod)
        mod.evaluate()

    def insert(self, idx, mod):
        """ Insert a module mod at index idx in stack, evaluate the inserted
        module, re-append the later modules, and evaluate starting at idx+1.

        Returns idx, the module index at which the module was inserted. It will
        either be the same as the argument idx if given, or the last index in
        the stack if mod was appended instead. """
        n = len(self.modules)
        if (not idx) or (idx > n-1) or (idx < -1*n-1):
            raise IndexError
        if idx == n:
            self.append(module)
            return len(self.modules)
        self.modules.insert(idx, module)
        self.evaluate(start_idx=idx)
        return idx

    def remove(self, idx=None, mod=None, all_idx=False):
        """ Remove the module at the given index in stack, then re-append
        all modules that came after the removed module.

        Arguments:
        ----------
        idx : int
            Index of module to be removed. If no idx is given, but a mod is
            given, find_modules() will be used to find idx.

        mod : nems_module
            Module to be removed. Only needs to be passed if index is not
            given, or if all matching indices should be removed.

        all_idx : boolean
            If true, and a module is passed instead without an idx,
            all matching instances of the module will be removed.
            Otherwise the instance at the highest index will be removed.

        Errors:
        -------
        Raises an IndexError if idx is None or its absolute value is greater
        than the length of self.modules, or if mod is given but not found.
        """
        n = len(self.modules)
        if (not mod and not idx) or (idx > n-1) or (idx < -1*n-1):
            raise IndexError

        idxs = find_modules(self, mod.name)
        if not idxs:
            raise ValueError("Module does not exist in stack.")

        if not all_idx:
            # Only remove the instance with the highest index
            self.remove(idx=idxs[-1], mod=None)
            return
        else:
            j = idxs[0]
            del self.modules[j]
            self.evaluate(start_idx=j-1)

    def pop(self):
        del self.modules[-1]

    def clear(self):
        del self.modules[:]

    def output(self):
        return self.modules[-1].data

    def fit(self):
        print('NOT YET IMPLEMENTED')
        return None

    # def quick_plot(self, size=(12, 24)):
    #     fig = plt.figure(figsize=size)

    #     # find all modules with plotfn
    #     plot_set = [m for m in self.modules if m.auto_plot_idx]

    #     # outer grid corresponding to a subplot for each of the modules:
    #     outer = gridspec.GridSpec(len(plot_set), 1)

    #     # For old subplot handling, since they have 1 based indexing:
    #     spidx = 1

    #     for sp, idx in enumerate(plot_set):
    #         print("quick_plot: {}".format(self.modules[idx].name))
        
      
    #         try:
    #             # if the module specific plotting uses an inner subplot grid, passes the outer grid
    #             # this implementation is quite nasty since it relies on an error to choose between cases
    #             # although passing the handlers of figure and outer grid to the plotting function
    #             # seems overall much more cleaner.
    #             self.modules[idx].do_plot(self.modules[idx], figure  =  fig ,outer = outer[sp]) #, wspace = 0.2, hspace = 0.2)
    #         except:
    #             mod_ax  =  plt.Subplot(fig, outer[sp])
    #             fig.add_subplot(mod_ax)
    #             self.modules[idx].do_plot(self.modules[idx])

    #         #if idx =  = plot_set[0]:
    #         #    plt.title("{0} - {1} - {2}".format(self.meta['cellid'],self.meta['batch'],self.meta['modelname']))
    #         spidx+ = 1

    #     fig.suptitle("{0} - {1} - {2}"
    #                  .format(self.meta['cellid'],
    #                          self.meta['batch'],
    #                          self.meta['modelname']))
        
    #     #plt.tight_layout()
    #     #TODO: Use gridspec to fix spacing issue? Addition of labels makes
    #     #      the subplots look "scrunched" vertically.
    
    # def quick_plot_save(self, mode = 'png'):
    #     """Copy of quick_plot for easy save or embed.
        
    #     mode options:
    #     -------------
    #     "json" -- .json
    #     "html" -- .html
    #     "png" -- .png
    #     default -- .png

    #     returns:
    #     --------
    #     filename : string
    #         Path to saved file, currently of the form:
    #         "/auto/data/code/nems_saved_images/batch{#}/{cell}/{modelname}.type"
    #     @author: jacob
    #     """
    #     batch  =  self.meta['batch']
    #     cellid  =  self.meta['cellid']
    #     modelname  =  self.meta['modelname']
    
    #     fig  =  plt.figure(figsize = (8,9))
    #     for idx,m in enumerate(self.modules):
    #     # skip first module
    #         if idx > 0:
    #             print(self.mod_names[idx])
    #             plt.subplot(len(self.modules)-1,1,idx)
    #             m.do_plot(m)
    #     plt.tight_layout()
    
    #     filename  =  (
    #                 '/tmp/' + "nems_saved_images/batch{0}/{1}/{2}.{3}"
    #                 .format(batch, cellid, modelname, mode)
    #                 )

    #     if AWS:
    #         s3  =  boto3.resource('s3')
    #         key  =  filename[len(sc.DIRECTORY_ROOT):]
    #         fileobj  =  io.BytesIO()
    #         fig.savefig(fileobj, format = mode)
    #         fileobj.seek(0)
    #         s3.Object(sc.PRIMARY_BUCKET, key).put(Body = fileobj)
    #         # return ("s3://" + sc.PRIMARY_BUCKET + "/" + key)
    #     else:
    #         dr  =  (
    #                 '/tmp/'
    #                 + "nems_saved_images/batch{0}/{1}/".format(batch, cellid)
    #                 )
    #         try:
    #             os.stat(dr)
    #             if os.path.isfile(filename):
    #                 os.remove(filename)
    #         except:
    #             os.mkdir(dr)

    #         try:
    #             fig.savefig(filename)
    #         except Exception as e:
    #             print("Bad file extension for figure or couldn't save")
    #             print(e)
    #             raise e

    #         try:
    #             os.chmod(filename, 0o666)
    #         except Exception as e:
    #             print("Couldn't modify file permissions for figure")
    #             raise e

    #     return filename

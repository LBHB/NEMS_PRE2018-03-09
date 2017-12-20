#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:54:26 2017

@author: shofer
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nems.utilities as ut
import numpy as np
import os
import io
import copy

try:
    import boto3
    import nems_config.Storage_Config as sc
    AWS = sc.USE_AWS
except BaseException:
    from nems_config.defaults import STORAGE_DEFAULTS
    sc = STORAGE_DEFAULTS
    AWS = False


class nems_stack:

    """
    Key components:
        modules: list of nems_modules in sequence of execution
        data: stream of data as it is evaluated through the sequence of modules
        fitter: pointer to the fit module
        quick_plot: generates a plot of something about the transformation
                     that takes place at each modules step
        nests: number validation sets to create. Used for nested crossvalidation.
                If 5% validation sets are wanted, use 20 nests, etc.
        parm_fits: The fitted parameters for each nest. A list of phi vectors.
        plot_dataidx: which dataset to plot (estimation or validation). Usually,
                        0 is the estimation set and 1 is the validation set, but
                        sometimes there are more sets. Even numbers (and zero)
                        are estimation sets, while odd numbers are validation sets.
        plot_stimidx: which stimulus to plot



    """

    # modelname=None
    modules = []  # stack of modules
    mod_names = []
    mod_ids = []
    data = []     # corresponding stack of data in/out for each module
    meta = {}
    fitter = None
    valmode = False
    nests = 0
    avg_resp = True
    plot_dataidx = 0
    plot_stimidx = 0
    parm_fits = []
    fitted_modules = []
    cv_counter = 0
    keywords = []
    keyfuns = []  # to be populated from nems.keywords.keyfuns - dumb hack

    def __init__(self, cellid="X", batch=0, modelname="X"):
        print("Creating new stack")
        self.modules = []
        self.mod_names = []
        self.data = []
        self.data.append([])
        self.data[0].append({})  # Also this?
        self.data[0][0]['resp'] = []  # Do we need these?
        self.data[0][0]['stim'] = []  # This one too?

        self.meta = {}  # Dictionary that will contain cellid, batch, modelname
        self.meta['cellid'] = cellid
        self.meta['batch'] = batch
        self.meta['modelname'] = modelname
        self.meta['nests'] = 0
        self.meta['cv_counter'] = 0

        # extract keywords from modelname
        self.keywords = modelname.split("_")

        # TODO: move to meta:
        self.nests = 0  # Default is to have only one nest, i.e. standard crossval
        self.cv_counter = 0  # Counter for iterating through nests, used in nm.crossval

        self.error = self.default_error
        self.valmode = False
        # If the data is resampled by load_mat, holds an unresampled copy for
        # raster plot
        self.unresampled = []
        self.parm_fits = []  # List of fitted parameters for each nest
        self.fitted_modules = []  # List of modules with fitted parameters
        self.mod_ids = []
        # self.valfrac=0.05 #Fraction of the data used to create each
        # validation nest

    def get_phi(self, module_subset=None):
        if module_subset:
            phi=[]
            for ii in range(0,len(self.modules)):
                if ii in module_subset:
                    phi.append(self.modules[ii].get_phi())
                else:
                    phi.append({})
            return phi
        else:
            return [module.get_phi() for module in self.modules]

    def set_phi(self, phi):
        for p, m in zip(phi, self.modules):
            m.set_phi(p)

    def evaluate(self, start=0):
        """
        Evaluate modules in stack, starting at module # start. When stack is
        fitting (valmode is False), simply calls mod.evaluate() for each module
        in the stack. However, when the stack is testing (valmode is True),
        evaluate first calls an est/val function to create a list of validation
        datasets ("nest"). It then extracts the fitted parameters for each list, and calls
        mod.evaluate for each module in the stack for each validation set in the
        list.

        Note that during valmode, the estimation dataset that is returned is the
        last dataset fit.
        """

        if self.valmode and self.nests > 0:
            # evaluate using the old nesting scheme devised by Noah
            # this section is deprecated and should be deleted

            print('Evaluating nested validation data')
            mse_idx = ut.utils.find_modules(self, 'metrics.mean_square_error')
            mse_idx = int(mse_idx[0])
            try:
                xval_idx = ut.utils.find_modules(self, 'est_val.crossval')[0]
            except BaseException:
                xval_idx = 1

            if start > xval_idx:
                start = xval_idx
            for ii in range(start, xval_idx + 1):
                self.modules[ii].evaluate()

            start = xval_idx + 1
            for ii in range(start, mse_idx):
                for cv_counter in range(0, self.nests):
                    print("Eval {0} in valmode, nest={1}".format(
                        ii, cv_counter))
                    st = 0
                    for m in self.fitted_modules:
                        phi_old = self.modules[m].parms2phi()
                        s = phi_old.shape
                        self.modules[m].phi2parms(
                            self.parm_fits[cv_counter][st:(st + np.prod(s))])
                        st += np.prod(s)
                    self.modules[ii].evaluate(nest=cv_counter)
            ut.utils.concatenate_helper(
                self, start=xval_idx + 1, end=mse_idx + 1)

            for ii in range(mse_idx, len(self.modules)):
                self.modules[ii].evaluate()

        elif self.valmode and self.meta['nests'] > 0:
            # new nested eval, doesn't require special evaluation procedure
            # by modules. instead, all the remapping and collecting of val
            # data from each nest handled inside this evaluation function.

            # 2017-09-08- still something not quite right.
            # 2017-10-06- working finally. this is new norm for cross-val
            stack = self

            try:
                xval_idx = ut.utils.find_modules(stack, 'est_val.crossval')[0]
            except BaseException:
                xval_idx = 1

            try:
                mse_idx = ut.utils.find_modules(
                    stack, 'metrics.mean_square_error')
                mse_idx = int(mse_idx[0])
            except BaseException:
                mse_idx = len(self.modules)

            print("Evaluating nested validation data: xvidx={0} mseidx={1}".format(
                xval_idx, mse_idx))

            # evaluate up to xval module (if necessary)
            if start > xval_idx:
                start = xval_idx
            for ii in range(start, xval_idx):
                print("eval {0} in valmode".format(ii))
                stack.modules[ii].evaluate()

            # go through each nest and evaluate stack, saving data stack to
            # a placeholder (d_save). copy all keys from data stack except
            # known metadata listed in exclude_keys. this allows for arbitrary
            # new data elements to be created (stim2, etc)
            exclude_keys = ['avgresp', 'poststim', 'fs', 'isolation', 'stimparam', 'filestate',
                            'prestim', 'duration', 'est', 'stimFs', 'respFs',
                            'resp_raw', 'pupil_raw']
            include_keys = {}
            d_save = {}
            for cv_counter in range(0, stack.meta['nests']):
                # for cv_counter in range(0,1):
                stack.meta['cv_counter'] = cv_counter

                # load in parameters for appropriate nested fit
                st = 0
                for midx in stack.fitted_modules:
                    phi_old = stack.modules[midx].parms2phi()
                    s = phi_old.shape
                    # stack.modules[midx].phi2parms(stack.parm_fits[cv_counter][st:(st+np.prod(s))])
                    stack.modules[midx].phi2parms(
                        stack.parm_fits[cv_counter][st:(st + np.prod(s))])
                    st += np.prod(s)

                # evaluate stack for this nest up to before error metric
                # modules
                for ii in range(xval_idx, mse_idx):
                    print("nest={0}, eval {1} in valmode".format(
                        cv_counter, ii))
                    stack.modules[ii].evaluate()

                    # append data from this (nest,module) onto d_save
                    # placeholders:
                    if cv_counter == 0:
                        include_keys[ii] = stack.modules[ii].d_out[0].keys(
                        ) - exclude_keys
                        d_save[ii] = copy.deepcopy(stack.modules[ii].d_out)
                        print(include_keys[ii])
                    for d, d2 in zip(stack.modules[ii].d_out, d_save[ii]):
                        if not d['est'] and cv_counter > 0:
                            for k in include_keys[ii]:
                                if d[k] is None or d[k] == []:
                                    pass
                                elif k == 'pupil' and d[k].ndim == 3 and d['resp'].shape[1] != d[k].shape[1]:
                                    d2[k] = np.append(d2[k], d[k], axis=2)
                                elif d[k].ndim == 3:
                                    d2[k] = np.append(d2[k], d[k], axis=1)
                                else:
                                    d2[k] = np.append(d2[k], d[k], axis=0)

            # save accumulated data stack back to main data stack

            # figure out mapping to get val data back into original order
            for ii in range(xval_idx, mse_idx):
                jj = -1
                for d, d2 in zip(stack.modules[ii].d_out, d_save[ii]):
                    if not d['est']:
                        jj += 1  # count validation files only
                        validx = np.concatenate(
                            stack.modules[xval_idx].validx_sets[jj])
                        mapidx = np.argsort(validx)
                        for k in include_keys[ii]:
                            if d2[k] is None or d[k] == []:
                                d[k] = d2[k]
                            elif k == 'pupil' and d[k].ndim == 3 and d2['resp'].shape[1] != d2[k].shape[1]:
                                d[k] = d2[k][:, :, mapidx]
                            elif d[k].ndim == 3:
                                d[k] = d2[k][:, mapidx, :]
                            elif d[k].ndim == 2:
                                d[k] = d2[k][mapidx, :]
                            else:
                                d[k] = d2[k][mapidx]

            # continue standard evaluation of later stack elements
            for ii in range(mse_idx, len(self.modules)):
                print("eval {0} in valmode".format(ii))
                self.modules[ii].evaluate()

        else:
            # standard evaluation when not using nested cross-validation
            for ii in range(start, len(self.modules)):
                self.modules[ii].evaluate()

    # create instance of mod and append to stack
    def append(self, mod=None, **xargs):
        """
        Creates an instance of a module and appends it to the stack. Evaluates
        module in doing so.
        """
        if mod is None:
            raise ValueError('stack.append: module not specified')
        else:
            m = mod(self, **xargs)
        self.append_instance(m)

    def append_instance(self, mod=None):
        """Same as append but takes an instance of a module instead
        of the class to preserve existing **xargs. For use with insert/remove.
        Could maybe merge these with an added boolean arg? Wasn't sure if that
        would interfere with the **xargs.

        @author: jacob

        """
        if not mod:
            raise ValueError('stack.append: module not specified')
            # mod=nm.nems_module(self)
        self.modules.append(mod)
        self.data.append(mod.d_out)
        self.mod_names.append(mod.name)
        self.mod_ids.append(mod.idm)

        mod.evaluate()

    def append_no_eval(self, mod=None, **xargs):
        """
        Creates an instance of a module and appends it to the stack. Evaluates
        module in doing so.

        APPEARS TO BE A BUG. Is input a module instance? Or a pointer to the function?
        """
        if mod is None:
            raise ValueError('stack.append: module not specified')
        else:
            m = mod(self, **xargs)
        self.modules.append(mod)
        self.data.append(mod.d_out)
        self.mod_names.append(mod.name)
        self.mod_ids.append(mod.idm)

    def insert(self, mod=None, idx=None, **xargs):
        """Insert a module at index in stack, then evaluate the inserted
        module and re-append all the modules that were in the stack previously,
        starting with the insertion index.

        Returns:
        --------
        idx : int
            Index the module was inserted at.
            Will either be the same as the argument idx if given, or
            the last index in the stack if mod was appended instead.

        @author: jacob

        """

        # if no index is given or index is out of bounds,
        # just append mod to the end of the stack
        if (not idx) or (idx > len(self.modules) - 1)\
                or (idx < -1 * len(self.modules) - 1):
            self.append(mod, **xargs)
            idx = len(self.modules) - 1
            return idx

        tail = [self.popmodule_2() for m in self.modules[idx:]]
        self.append(mod, **xargs)
        # originally tail[:-1], excluding last module. not sure if this is for
        # Nested
        for mod in reversed(tail[:]):
            self.append_instance(mod)
        return idx

    def remove(self, idx=None, mod=None, all_idx=False):
        """Remove the module at the given index in stack, then re-append
        all modules that came after the removed module.

        Arguments:
        ----------
        idx : int
            Index of module to be removed. If no idx is given, but a mod is
            given, ut.utils.find_modules will be used to find idx.
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

        @author: jacob

        """

        if mod and not idx:
            idx = ut.utils.find_modules(self, mod.name)
            if not idx:
                print("Module does not exist in stack.")
                return
            if not all_idx:
                # Only remove the instance with the highest index
                self.remove(idx=idx[-1], mod=None)
                return
            else:
                j = idx[0]
                # same as tail comp below, but exclude all indexes that match
                # the mod
                tail_keep = [
                    self.popmodule_2() for i, m in
                    enumerate(self.modules[j:])
                    if i not in idx
                ]
                # still need to pop the matched instances, but don't need to
                # do anything with them.
                tail_toss = [
                    self.popmodule_2() for i, m in
                    enumerate(self.modules[j:])
                    if i in idx
                ]
                for mod in reversed(tail_keep):
                    self.append_instance(mod)

        if (not idx) or (idx > len(self.modules) - 1)\
                or (idx < -1 * len(self.modules) - 1):
            raise IndexError

        # Remove modules from stack starting at the end until reached idx.
        tail = [self.popmodule_2() for m in self.modules[idx:]]
        # Then put them back on starting with the second to last module popped.
        for mod in reversed(tail[:-1]):
            self.append_instance(mod)

    def popmodule_2(self):
        """For remove and insert -- wasn't sure if the original popmodule
        method had a specific use-case, so I didn't want to modify it.
        Can merge the two if these changes won't cause issues.

        Removes the last module from stack lists, along with its corresponding
        data and name (and id?).

        @author: jacob
        """

        m = self.modules.pop(-1)
        self.data.pop(-1)
        self.mod_names.pop(-1)
        # Doesn't look like this one is being used yet?
        # self.mod_ids.pop(-1)
        return m

    def popmodule(self):
        del self.modules[-1]
        del self.data[-1]

    def clear(self):
        del self.modules[:]
        del self.data[1:]

    def output(self):
        return self.data[-1]

    def default_error(self):
        return np.zeros([1, 1])

    def quick_plot(self, size=(12, 24)):
        fig = plt.figure(figsize=size)

        # find all modules with plotfn
        plot_set = []
        for idx, m in enumerate(self.modules):
            if m.auto_plot:
                plot_set.append(idx)
        # outer grid corresponding to a subplot for each of the modules.
        outer = gridspec.GridSpec(len(plot_set), 1)

        # this is for old subplot handling, since they have 1 based indexing.
        spidx = 1
        for sp, idx in enumerate(plot_set):
            print("quick_plot: {}".format(self.modules[idx].name))

            # plt.subplot(len(plot_set),1,spidx)   <- this is to work without
            # grispec
            try:
                # if the module specific plotting uses an inner subplot grid, passes the outer grid
                # this implementation is quite nasty since it relies on an error to choose between cases
                # although passing the handlers of figure and outer grid to the plotting function
                # seems to be what matplotlib wants us to do .
                self.modules[idx].do_plot(self.modules[idx], figure = fig ,outer=outer[sp]) #, wspace=0.2, hspace=0.2)
            except BaseException:
                mod_ax = plt.Subplot(fig, outer[sp])
                fig.add_subplot(mod_ax)
                self.modules[idx].do_plot(self.modules[idx])

            # if idx==plot_set[0]:
            #    plt.title("{0} - {1} - {2}".format(self.meta['cellid'],self.meta['batch'],self.meta['modelname']))
            spidx += 1

        fig.suptitle(
            "{0} - {1} - {2}".format(self.meta['cellid'], self.meta['batch'], self.meta['modelname']))

        # plt.tight_layout()
        # TODO: Use gridspec to fix spacing issue? Addition of labels makes
        #      the subplots look "scrunched" vertically.

    def quick_plot_save(self, mode='png'):
        """Copy of quick_plot for easy save or embed.

        mode options:
        -------------
        "json" -- .json
        "html" -- .html
        "png" -- .png
        default -- .png

        returns:
        --------
        filename : string
            Path to saved file, currently of the form:
            "/auto/data/code/nems_saved_images/batch{#}/{cell}/{modelname}.type"

        @author: jacob

        """
        batch = self.meta['batch']
        cellid = self.meta['cellid']
        modelname = self.meta['modelname']

        fig = plt.figure(figsize=(8, 9))
        for idx, m in enumerate(self.modules):
            # skip first module
            if idx > 0:
                print(self.mod_names[idx])
                plt.subplot(len(self.modules) - 1, 1, idx)
                m.do_plot(m)
        plt.tight_layout()

        filename = (
            sc.DIRECTORY_ROOT + "nems_saved_images/batch{0}/{1}/{2}.{3}"
            .format(batch, cellid, modelname, mode)
        )

        if AWS:
            s3 = boto3.resource('s3')
            key = filename[len(sc.DIRECTORY_ROOT):]
            fileobj = io.BytesIO()
            fig.savefig(fileobj, format=mode)
            fileobj.seek(0)
            s3.Object(sc.PRIMARY_BUCKET, key).put(Body=fileobj)
            # return ("s3://" + sc.PRIMARY_BUCKET + "/" + key)
        else:
            dr = (
                sc.DIRECTORY_ROOT
                + "nems_saved_images/batch{0}/{1}/".format(batch, cellid)
            )
            try:
                os.stat(dr)
                if os.path.isfile(filename):
                    os.remove(filename)
            except BaseException:
                os.mkdir(dr)

            try:
                fig.savefig(filename)
            except Exception as e:
                print("Bad file extension for figure or couldn't save")
                print(e)
                raise e

            try:
                os.chmod(filename, 0o666)
            except Exception as e:
                print("Couldn't modify file permissions for figure")
                raise e

        return filename

#!/usr/bin/python3

import os
import sys
import numpy as np
import scipy.io
from nems.Signal import Signal


###############################################################################
# For unpacking matlab stuff

def print_usage():
    print("""Usage: mat2csv <MATFILE> [<MATFILE> ...]""")

def unwrap(n):
    if type(n) == np.ndarray:
        return n[0]
    else:
        return n

def fix_np_type(m):
    """ Convert 1x1 numpy datatypes into standard python data types. """
    if isinstance(m, np.str_):
        return m
    elif isinstance(m, np.ndarray):
        if m.shape == (1,1):
            return fix_np_type(m[0, 0])
        else:
            return m
    elif isinstance(m, np.integer):
        return int(m)
    elif isinstance(m, np.floating):
        return float(m)
    elif isinstance(m, np.void):
        return None
    else:
        raise ValueError(str("Unhandled type:")+str(type(m)))

def __extract_metadata(matrec):
    """ Extracts metadata from the matlab matrix object matrec. """
    found = set(matrec.dtype.names)
    needed = ['cellid', 'isolation', 'stimfs', 'respfs',
              'stimchancount', 'stimfmt', 'filestate']
    meta = dict((n, unwrap(matrec[n][0])) for n in needed if n in found)

    for k, v in meta.items():
        meta[k] = fix_np_type(v)

    meta['cellid'] = str(meta['cellid'])  # Convert numpy string to normal

    # Tags is not completely examined here; TODO: find others?
    meta['prestim'] = float(matrec['tags'][0]['PreStimSilence'][0][0][0])
    meta['poststim'] = float(matrec['tags'][0]['PostStimSilence'][0][0][0])
    meta['duration'] = float(matrec['tags'][0]['Duration'][0][0][0])

    # TODO: fn_spike, fn_param metadata are mostly ignored; too complicated!
    meta['stimparam'] = [str(''.join(letter)) for letter in matrec['fn_param']]

    # TODO: These are not metadata so much as 'baked-in' model fitting
    # parameters. Remove anywhere found.
    # meta['est'] = matrec['estfile']
    # meta['repcount']=np.sum(np.isfinite(data['resp'][0,:,:]),axis=0)

    return meta

def mat2signals(matfile):
    """ Converts a matlab file into a set of signals. """
    matdata = scipy.io.loadmat(matfile,
                               chars_as_strings=True,
                               squeeze_me=False)

    # Remove useless variables
    del matdata['__header__']
    del matdata['__globals__']
    del matdata['__version__']

    found_matrices = set(matdata.keys())

    toplevel_matrices = {}
    for m in sorted(found_matrices):
        d = matdata[m]
        if type(d) is np.ndarray and d.shape == (1,1):
            toplevel_matrices[m] = fix_np_type(d)
            found_matrices.remove(m)

    for k in found_matrices:
        print(k, matdata[k].shape, type(matdata[k]))

    # Verify that .mat file has no unexpected matrix variables
    expected_matlab_vars = set(['data', 'cellid'])
    if not found_matrices == expected_matlab_vars:
        raise ValueError("Unexpected variables found in .mat file: "
                         + str(sorted(found_matrices)))

    sigs = []
    for m in matdata['data'][0, :]:
        meta = __extract_metadata(m)
        meta['recording'] = os.path.basename(meta['stimparam'][0])

        # Verify that this is not a 'stimid' recording file format
        signal_names = set(m.dtype.names)
        if 'stimids' in signal_names:
            raise ValueError('stimids are not supported yet, sorry')

        # Extract the two required signals
        sigs.append(Signal(signal_name='stim',
                           recording=meta['recording'],
                           meta={k: meta[k] for k in meta
                                 if k in set(['duration',
                                              'prestim',
                                              'poststim',
                                              'stimfmt',
                                              'stimchancount',
                                              'filestate'])},
                           matrix=np.swapaxes(m['stim'], 0, 1),
                           fs=meta['stimfs']))

        guessed_cellid = meta['cellid'][-3:]
        if guessed_cellid[0] != '-':
            raise ValueError("I think I guessed the wrong cellid suffix!")
        sigs.append(Signal(signal_name='resp' + guessed_cellid,
                           recording=meta['recording'],
                           meta={k: meta[k] for k in meta
                                 if k in set(['isolation'])},
                           matrix=m['resp_raster'],
                           fs=meta['respfs']))

        # Extract the pupil size and behavior_condition, if they exist
        if 'pupil' in signal_names:
            sigs.append(Signal(signal_name='pupil',
                               recording=meta['recording'],
                               meta=None,
                               matrix=m['pupil']*0.01,
                               fs=meta['respfs']))

        # TODO: instead of respfs, switch to pupilfs and behavior_conditionfs
        if 'behavior_condition' in signal_names:
            sigs.append(Signal(signal_name='behavior_condition',
                               recording=meta['recording'],
                               meta=None,
                               matrix=np.swapaxes(m['behavior_condition'], 0, 1),
                               fs=meta['respfs']))

    return sigs


###############################################################################
# SCRIPT STARTS HERE

if len(sys.argv) < 2:
    print_usage()
    exit(-1)

matfiles = sys.argv[1:]

for f in matfiles:
    print(f)
    sigs = mat2signals(f)
    for s in sigs:
          print("\t", s.recording, s.name, s.__matrix__.shape, s.meta)
          (csv, js) = s.savetocsv('./', fmt='%1.5e')


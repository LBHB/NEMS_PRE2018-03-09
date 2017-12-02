import os
import numpy as np
import scipy.io
from nems.Signal import Signal, loadfromcsv

DEFAULT_DIRPATH = '/home/ivar/mat/'


###############################################################################
# For unpacking matlab stuff

def __extract_metadata(matrec):
    """ Extracts metadata from the matlab matrix object matrec. """
    found = set(matrec.dtype.names)
    needed = ['cellid', 'isolation', 'stimfs', 'respfs',
              'stimchancount', 'stimfmt', 'filestate']
    unwrap = lambda n: n[0] if type(n) == np.ndarray else n
    meta = dict((n, unwrap(matrec[n][0])) for n in needed if n in found)

    # Convert into integers or floats cuz np datatypes are not serializable
    for k, v in meta.items():
        if isinstance(v, np.str_):
            next
        elif isinstance(v, np.integer):
            meta[k] = int(v)
        elif isinstance(v, np.floating):
            meta[k] = float(v)
        else:
            print("WARN:", k, 'is', type(v))

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

    # Verify that .mat file has no unexpected matrix variables
    expected_matlab_vars = set(['data', '__globals__', '__version__',
                                '__header__', 'cellid'])
    found_matrices = set(matdata.keys())
    if not found_matrices == expected_matlab_vars:
        raise ValueError("Unexpected variables found in .mat file: "
                         + found_matrices)

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

matfile = ('/home/ivar/tmp/gus027b-a1_b293_parm_fs200'
           + '/gus027b-a1_b293_parm_fs200.mat')

sigs = mat2signals(matfile)

print("---")
for s in sigs:
    print(s.recording, s.name, s.__matrix__.shape, s.meta)
    (csv, js) = s.savetocsv('/home/ivar/sigs/', fmt='%1.5e')
    q = loadfromcsv(csv, js)
    print(q.recording, q.name, q.__matrix__.shape, q.meta)

##############################################################################

# A Test of the above
with open('/tmp/ooo.csv', 'w') as f:
    n = 0
    for i in range(200):
        f.write(str(n))
        n += 1
        for j in range(1, 8):
            f.write(', ')
            f.write(str(n))
            n += 1
        f.write('\n')

q = loadfromcsv('/tmp/ooo.csv', '/tmp/ooo.json')
q.savetocsv('/tmp/')

assert(q.__matrix__[1,2,3] == 490)
assert(q.as_average_trial()[3,3] == 747.0)

p = q.jackknifed_by_reps(10, 0)
assert(np.isnan(p.__matrix__[1,3,0]))
assert(np.isnan(p.__matrix__[2,5,0]))
assert(not np.isnan(p.__matrix__[2,5,1]))

r = q.jackknifed_by_time(200, 199)
assert(np.isnan(r.__matrix__[-1,3,-1]))
assert(not np.isnan(r.__matrix__[1,3,1]))

#print(r.__matrix__)
#print(r.as_single_trial())

print("Tests passed")

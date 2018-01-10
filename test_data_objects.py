import os
import json
import filecmp
import pytest
from nems.signal import Signal
# import nems.recording


def generate_dummy_signal_files(tmpdir, 
                                signal_name='dummy_signal',
                                recording_name='dummy_recording',
                                fs=50,
                                nchans=3,
                                ntimes=200,
                                nreps=8):
    '''
    Generates dummy signal file with a predictable structure (every element
    increases by 1) that is useful for testing. Returns 'basepath' of the 
    signal, which is the filepath without the .json or .csv extension.
    '''
    basepath = os.path.join(str(tmpdir), signal_name)

    with open(basepath+'.csv', 'w') as f:
        n=0
        for _ in range(ntimes):
            f.write('{:1.3e}'.format(float(n)))
            n += 1
            for _ in range(nchans):
                f.write(', {:1.3e}'.format(float(n)))
                n += 1
            f.write('\n')

    metadata = {'name': signal_name,
                'recording': recording_name,
                'chans': ['chan' + str(n) for n in range(nchans)],
                'fs': fs,
                'nreps': nreps,
                'meta': {'for_testing': True, 
                         'date': "2018-01-10",
                         'animal': "Donkey Hotey",                         
                         'windmills': 'tilting'}}

    with open(basepath+'.json', 'w') as f:
        json.dump(metadata, f)

    return basepath

# Create a tmp directory with a signal in it, test loading/saving
# of signals, and return an example siignal object for other tests
@pytest.fixture(scope='module')
def example_signal_object(tmpdir_factory):
    '''
    Test that signals object load/save methods work as intended, and
    return an example signal object for other tests.
    '''
    tmpdir = tmpdir_factory.mktemp("blah")

    print("Generating test signals...")
    basepath = generate_dummy_signal_files(tmpdir)

    print("Loading signal...")
    sig = Signal.load(basepath)

    print("Saving signal...")
    saved_directory = os.path.join(str(tmpdir), 'saved')
    os.mkdir(saved_directory)
    sig.save(saved_directory, fmt='%1.3e')

    print("Testing saved signal matches original...")
    sigs_found = Signal.list_signals(saved_directory)
    assert(len(sigs_found) == 1)
    basepath2 = os.path.join(saved_directory, sigs_found[0])
    f1 = basepath + '.csv'
    f2 = basepath2 + '.csv'
    files_are_same = filecmp.cmp(f1, f2)
    if files_are_same:
        assert files_are_same
    else:
        with open(f1, 'r') as f:
            first_line1 = f.readline()
        with open(f2, 'r') as f:
            first_line2 = f.readline()
        print(first_line1)
        print(first_line2)
        assert files_are_same

    return sig


def test_as_single_trial(example_signal_object):
    sig = example_signal_object
    assert(sig.as_single_trial().shape == (200, 3))

def test_as_average_trial(example_signal_object):
    sig = example_signal_object
    assert(sig.as_average_trial().shape == (25, 3))

def test_as_repetition_matrix(example_signal_object):
    sig = example_signal_object
    assert(sig.as_single_trial().shape == (25, 8, 3))

def test_normalized():

    # m = sig.normalized().as_single_trial()
    print('Done!')


#@pytest.fixture
#def test_signal(tmpdir)

# +print("---")
# +for s in sigs:
# +    print(s.cellid, s.recording, s.name, s.__matrix__.shape, s.meta)
# +    (csv, js) = s.savetocsv('/home/ivar/sigs/', fmt='%1.5e')
# +    q = loadfromcsv(csv, js)
# +    print(q.cellid, q.recording, q.name, q.__matrix__.shape, q.meta)



#  # A Test of the above

 
# -q = nems.Signal.loadfromcsv('/tmp/ooo.csv', '/tmp/ooo.json')
# +q = loadfromcsv('/tmp/ooo.csv', '/tmp/ooo.json')
#  q.savetocsv('/tmp/')
 
#  assert(q.__matrix__[1,2,3] == 490)
# +assert(q.as_average_trial()[3,3] == 747.0)
# +
# +p = q.jackknifed_by_reps(10, 0)
# +assert(np.isnan(p.__matrix__[1,3,0]))
# +assert(np.isnan(p.__matrix__[2,5,0]))
# +assert(not np.isnan(p.__matrix__[2,5,1]))
# +
# +r = q.jackknifed_by_time(200, 199)
# +assert(np.isnan(r.__matrix__[-1,3,-1]))
# +assert(not np.isnan(r.__matrix__[1,3,1]))
# +
# +#print(r.__matrix__)
# +#print(r.as_single_trial())
# +
# +print("Tests passed")

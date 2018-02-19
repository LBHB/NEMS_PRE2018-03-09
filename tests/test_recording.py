import nems.epoch as ep
from nems.recording import Recording

# TODO: Make this into a unit test:

# rec = Recording.load(os.path.join(signals_dir, 'TAR010c-57-1'))
# est, val = rec.split_using_epoch_occurrence_counts(epoch_regex='^STIM_')

# print(ep.group_epochs_by_occurrence_counts(est['stim'].epochs, regex='^STIM_'))
#val_epochs = ['STIM_00Oxford_male2b.wav', 'STIM_00ferretmixed41.wav', 'STIM_00ferretmixed42.wav']
#mat = val['stim'].select_epochs(val_epochs).as_continuous()
#print('Non-NaN elements: {}'.format(np.count_nonzero(mat)))
#print('Total elements:   {}'.format(mat.size))

# assert(np.count_nonzero(mat) == mat.size)

# This should (and does!) throw an exception because they are not in the est set.
# print(est['stim'].select_epochs(val_epochs))

#!/usr/bin/python3

import sys
import nems.xforms as xforms
import nems.urls as urls


def fit_model(recording_uri, modelstring, destination):
    '''
    Fit a single model and save it to nems_db.
    '''
    recordings = [recording_uri]

    xfspec = [
        ['nems.xforms.load_recordings', {'recording_uri_list': recordings}],
        ['nems.xforms.add_average_sig', {'signal_to_average': 'resp',
                                         'new_signalname': 'resp',
                                         'epoch_regex': '^STIM_'}],
        ['nems.xforms.split_by_occurrence_counts', {'epoch_regex': '^STIM_'}],
        ['nems.xforms.init_from_keywords', {'keywordstring': modelstring}],
        ['nems.xforms.set_random_phi',  {}],
        ['nems.xforms.fit_basic',       {}],
        # ['nems.xforms.add_summary_statistics',    {}],
        # ['nems.xforms.plot_summary',    {}],
        # ['nems.xforms.save_recordings', {'recordings': ['est', 'val']}],
        ['nems.xforms.fill_in_default_metadata',    {}],
    ]

    ctx = xforms.evaluate(xfspec)

    urls.save_to_nems_db(destination,
                         modelspecs=ctx['modelspecs'],
                         xfspec=xfspec,
                         images=[],  # No images yet; put in ctx['images'] later
                         log=ctx['log'])


def print_usage():
    print('''
Usage:
      ./fit_model <recording> <modelkwstring> <destination>

Examples of valid arguments:
       <recording>      http://potoroo/recordings/TAR010c-02-1.tar.gz
       <recording>      file:///home/ivar/recordings/
       <modelkwstring>  wc18x1_lvl1_fir15x1
       <modelkwstring>  wc18x1_lvl1_fir15x1_dexp1
       <destination>    http://potoroo/recordings/
       <destination>    file:///home/ivar/recordings/
 ''')

# Parse the command line arguments and do the fit
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print_usage()
    else:
        fit_model(sys.argv[1], sys.argv[2], sys.argv[3])

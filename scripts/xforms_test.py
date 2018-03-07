# Ivar tries out the xforms concept, similar to how NARF used to work

import nems.xforms as xforms
import nems.urls as urls

recordings = ['http://potoroo/recordings/TAR010c-02-1.tar.gz']

modelstring = 'wc18x1_lvl1_fir15x1_dexp1'

destination = 'http://potoroo/results'

# This xfspec is built manually, but it could be done with 'xf keywords'
# analogous to how modelspecs are built from modelspec keywords.
#
# It is an explicit list of all operations and arguments given; this would
# have been easier in a LISP, but this style will have to do for Python.
#
# It's debatable is this easier to read than a python script, but it is
# probably more easily composed and rearranged than unstructured python.
#
# Also, we can save it as a JSON to where the modelspec is also saved.

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

# Ivar tries out the xforms concept, similar to how NARF used to work

import nems.xforms as xforms

recordings = ['http://potoroo/recordings/TAR010c-02-1.tar.gz']


xfspec = [{'xf': 'nems.xforms.load_recordings',
           'xf_kwargs': {'recording_uri_list': recordings}}]
#          {'xf': ,
#           'xf_kwargs': {}}]

ctx = xforms.evaluate(xfspec)

print(ctx)

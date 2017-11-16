""" For Jerb Central repository. """

import os
import binascii
import datetime
import json
import shutil
import uuid
import subprocess as sub
from jerb.Jerb import Jerb
from jerb.JerbRepo import JerbRepo

TMP_DIR = '/tmp/'
DEFAULT_PATH = '/home/ivar/central/'

class JerbCentral():
    def __init__(self, repopath=DEFAULT_PATH, create=False):
        self.JerbRepo = JerbRepo(repopath, create, bare=True)

    def absorb_jerb(self, jerb_to_absorb):
        """ Absorbs the jerb into the JerbCentral collective borg cube.
        The unique history of the Jerb will be lost, but its contents will
        remain integrated into the git repo."""
        my_uuid = str(uuid.uuid4())
        temp_repo_path = os.path.join(TMP_DIR, my_uuid)
        jr = JerbRepo(temp_repo_path, create=True)
        jr.unpack_jerb(jerb_to_absorb)
        sub.call(['git', 'update-ref', 'master', jerb_to_absorb.jid],
                 cwd=jr.repopath)
        sub.call(['git', 'checkout', 'master'],
                 cwd=jr.repopath)
        sub.call(['git', 'fetch', jr.repopath, '--quiet'],
                 cwd=self.JerbRepo.repopath)
        shutil.rmtree(temp_repo_path)

    def emit_jerb(self, jid):
        """ Emits a jerb found at JID. """
        my_uuid = str(uuid.uuid4())
        temp_repo_path = os.path.join(TMP_DIR, my_uuid)
        jr = JerbRepo(temp_repo_path, create=True)
        sub.call(['git', 'clone', '--depth=1', '--quiet',
                  self.JerbRepo.repopath, jid],
                 cwd=jr.repopath)
        js = jr.as_json()
        shutil.rmtree(temp_repo_path)
        return js

""" For Jerb Central repository. """

import os
import binascii
import datetime
import json
import shutil
import uuid
import subprocess as sub
from jerb.Jerb import Jerb
from jerb.JerbRepo import JerbRepo, PACK_DIR

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
        # TODO: get the metadata
        # TODO: Don't update master, update the correct branch
        with open('/dev/null', 'wb') as devnull:
            sub.call(['git', 'update-ref', 'master', jerb_to_absorb.jid],
                     cwd=jr.repopath)
            # It's weird, but I needed two checkouts:
            sub.call(['git', 'checkout', 'master'],
                     cwd=jr.repopath)
            sub.call(['git', 'checkout', '-b', 'master'],
                     stdout=devnull,
                     cwd=jr.repopath)
            sub.call(['git', 'fetch', jr.repopath, '--quiet'],
                     cwd=self.JerbRepo.repopath)
        shutil.rmtree(temp_repo_path)

    def emit_jerb(self, jid):
        """ Emits a jerb found at JID. """
        # TODO: Check that JID actually exists in repo already
        my_uuid = str(uuid.uuid4())
        temp_repo_path = os.path.join(TMP_DIR, my_uuid)
        sub.call(['git', 'branch', my_uuid, jid],
                 cwd=self.JerbRepo.repopath)
        sub.call(['git', 'clone', '-b', my_uuid, '--no-local', '--depth=1',
                  self.JerbRepo.repopath, temp_repo_path])
        sub.call(['git', 'branch', 'master'], cwd=temp_repo_path)
        sub.call(['git', 'checkout', 'master'], cwd=temp_repo_path)
        sub.call(['git', 'branch', '-d', my_uuid], cwd=temp_repo_path)
        md = sub.check_output(['git', 'log', '--pretty=format:%s',
                               jid], cwd=temp_repo_path)
        md = md.decode()
        md = json.loads(md)
        # Basically the same repack as in JerbRepo.py:
        sub.call(['git', 'repack', '-a', '-d', '--quiet'],
                 cwd=temp_repo_path)
        tmppck_dir = os.path.join(temp_repo_path, PACK_DIR)
        packs = [f for f in os.listdir(tmppck_dir) if f.endswith(".pack")]
        if 1 != len(packs):
            raise ValueError('More than one .pack file found:\n')
        with open(os.path.join(tmppck_dir, packs[0]), 'rb') as f:
            contents = f.read()
        # Create the object; unfortunately 'pack' will not be identical
        od = {'jid': jid,
              'meta': md,
              'pack': binascii.b2a_base64(contents).decode()}
        js = json.dumps(od, sort_keys=True)

        # Cleanup
        print('cleanup')
        shutil.rmtree(temp_repo_path)
        sub.call(['git', 'branch', '-d', my_uuid],
                 cwd=self.JerbRepo.repopath)
        return js

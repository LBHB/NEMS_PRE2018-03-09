#!/usr/bin/python

# Tries to make a Jerb file from this git repo's latest commit to master
# Please run in repo's base directory.

import os
import sys
import binascii
import json
import shutil
import uuid
import subprocess
# from git import Repo

TMP_DIR = '/tmp/'
PACK_DIR = '.git/objects/pack/'
MASTER_REF_PATH = '.git/refs/heads/master'


def ragequit(mesg):
    sys.stderr.write(mesg)
    sys.stderr.flush()
    sys.exit(-1)


def ensure_in_git_dir():
    if not os.path.isdir('./.git/'):
        ragequit('Error: Not in the base directory of a git repo.\n')


def make_single_pack():
    """ Makes a shallow clone of this repository's last commit only,
    then pack all it's git object files into a single .pack file """
    temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
    subprocess.call(['git', 'clone',
                     '--quiet',
                     '--no-local',
                     '--depth', '1',
                     '--branch', 'master',
                     '.', temp_repo_path])

    # Now repack the clone into a single .pack,
    # read out the pack,
    # then delete the clone repo
    subprocess.call(['git', 'repack', '-a', '-d', '--quiet'],
                    cwd=temp_repo_path)
    tmppck_dir = os.path.join(temp_repo_path, PACK_DIR)
    packs = [file for file in os.listdir(tmppck_dir) if file.endswith(".pack")]

    with open(os.path.join(tmppck_dir, packs[0]), 'rb') as f:
        pack = f.read()
    shutil.rmtree(temp_repo_path)

    # Rage quit if there were more than one pack files, because we screwed up!
    if 1 != len(packs):
        ragequit('Error: More than one .pack file found:\n')

    return pack


def get_master_ref():
    """ Returns the commit hash of the master ref. """
    with open(MASTER_REF_PATH, 'rb') as f:
        master_ref = f.read()
    return master_ref.decode().rstrip('\n')


def make_jerbstring(packfile, master_ref):
    """ Return a string containing a JSON Jerb string. """
    od = {'jid': master_ref,
          'meta': {"user": "ivar",
                   "key": "sampledata"},
          'pack': binascii.b2a_base64(packfile).decode()}
    j = json.dumps(od, sort_keys=True)
    return j


###############################################################################
# Script begins here:

ensure_in_git_dir()
pk = make_single_pack()
ref = get_master_ref()
js = make_jerbstring(pk, ref)

print(js)

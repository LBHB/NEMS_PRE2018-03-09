#!/usr/bin/python

# Unpacks one or more jerbs into the current git repository
# Usage:
#   unjerb <jerb1> [jerb2] [jerb3] ...

import os
import sys
import json
import binascii
import subprocess
import uuid
import tempfile

from jerb.Jerb import Jerb, load_jerb_from_file


TMP_DIR = tempfile.gettempdir()
PACK_DIR = '.git/objects/pack/'
MASTER_REF_PATH = '.git/refs/heads/master'


def ragequit(mesg):
    sys.stderr.write(mesg)
    sys.stderr.flush()
    sys.exit(-1)


def ensure_in_git_dir():
    if not os.path.isdir('./.git/'):
        ragequit('Error: Not in the base directory of a git repo.\n')


def add_local_jerbs():
    """ Reads in the arguments from the command line """
    # Ensure that all jerbs are actually real, loadable jerbs
    jerbs = []
    for j in sys.argv[1:]:
        if not os.path.exists(j):
            ragequit('Not a file: ' + j + "\n")
        jerbs.append(load_jerb_from_file(j))
    return jerbs


def unpack_jerb(jerb):
    """ Unpack the jerb into the current git repo as files (without adding
    them to the repo's next commit). """
    with open('/dev/null', 'w') as devnull:
        subprocess.run(['git', 'index-pack', '--stdin', '--keep'],
                       input=jerb.pack, stdout=devnull)
    print('Merging '+jerb.jid)
    subprocess.call(['git', 'merge', jerb.jid, '-q'])


###############################################################################
# Script begins here:

ensure_in_git_dir()

jerbs = add_local_jerbs()

for j in jerbs:
    unpack_jerb(j)


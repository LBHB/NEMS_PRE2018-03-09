""" jerb/shell.py: For shell and terminal interaction """

import os
import sys


def ragequit(mesg):
    """ Quit immediatley with an error message and a bad status code. """
    sys.stderr.write(mesg)
    sys.stderr.write('\n')
    sys.stderr.flush()
    sys.exit(-1)


def ensure_in_git_dir():
    if not os.path.isdir('.git/'):
        ragequit('Error: Not in the base directory of a git repo.\n')


def ensure_not_in_git_dir():
    if os.path.isdir('.git/'):
        ragequit('Error: Already in a git repo.\n')

import os
import sys
import binascii
import json
import shutil
import uuid
import subprocess

# TODO: Ensure that the git repo or environment variables have JERB_USER
# So that commits are done using the correct jerb system user name
# TODO: Use proper temporary directories


# Constants
TMP_DIR = '/tmp/'
PACK_DIR = '.git/objects/pack/'
MASTER_REF_PATH = '.git/refs/heads/master'


def ragequit(mesg):
    """ Dev use. Quit with an error message and a bad status code. """
    sys.stderr.write(mesg)
    sys.stderr.flush()
    sys.exit(-1)


def ensure_in_git_dir():
    if not os.path.isdir('.git/'):
        ragequit('Error: Not in the base directory of a git repo.\n')


def make_single_pack():
    """ Makes a shallow clone of this repository's last master commit only,
    then pack all it's git object files into a single .pack file """
    temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
    subprocess.call(['git', 'clone',
                     '--quiet',
                     '--no-local',
                     '--depth', '1',
                     '--branch', 'master',
                     '.', temp_repo_path])

    # Also fetch the notes
    subprocess.call(['git', 'fetch',
                     '--quiet',
                     '--depth', '1',
                     'origin', 'refs/notes/*:refs/notes/*'],
                    cwd=temp_repo_path)

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


def default_metadata():
    """ Try to create automatic metadata from the current git dir. """
    name = subprocess.check_output(['git', 'config', '--get', 'user.name'])
    email = subprocess.check_output(['git', 'config', '--get', 'user.email'])
    branch = os.path.basename(os.getcwd())
    md = {'user.name': name.strip().decode(),
          'user.email': email.strip().decode(),
          'branch': branch,
          'parents': '',
          'tags': '',
          'description': ''}
    return md


def add_parent_metadata(jid):
    """ Adds the jid to the parent metadata, if any exists. """

    # Get the existing metadata as a dictionary d
    if os.path.isfile('.git/jerb_metadata'):
        s = subprocess.check_output(['git', 'notes', 'show', 'jerb_metadata'])
        s = s.strip().decode()
        d = json.loads(s)
    else:
        # Ensure the 'hidden' repo metadata object exists in .git
        h = subprocess.check_output(['git', 'hash-object', '--stdin', '-w'],
                                    input='jerb_metadata'.encode())
        h = h.strip().decode()
        h = subprocess.run(['git', 'update-ref', 'jerb_metadata', h])
        d = default_metadata()

    # Add the JID to the "parents" list
    if 'parents' in d:
        parents = set([x.strip() for x in d['parents'].split(',')])
        parents.add(jid)
        d['parents'] = parents.join(', ')
    else:
        d['parents'] = jid

    # Save the updated json as a note
    js = json.dumps(d)
    subprocess.call(['git', 'notes', 'add', '-q', '-f', '-m', js,
                     'jerb_metadata'])


def unpack_jerb(jerb):
    """ Unpacks the jerb into the current git repo as files (without adding
    them to the repo's next commit). Also, adds the git commit hash of this
    jerb to the 'parents' metadata."""
    with open('/dev/null', 'w') as devnull:
        subprocess.run(['git', 'index-pack', '--stdin', '--keep'],
                       input=jerb.pack, stdout=devnull)
    subprocess.call(['git', 'merge', jerb.jid, '--quiet', '--no-edit'])
    add_parent_metadata(jerb.jid)

import os
import sys
import binascii
import json
import re
import shutil
import uuid
import subprocess
from jerb.Jerb import Jerb

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


def ensure_not_in_git_dir():
    if os.path.isdir('.git/'):
        ragequit('Error: It is bad practice to nest git repos.\n')


def is_SHA1_string(sha):
    """ Predicate. True when S is a valid SHA1 string."""
    r = re.compile('^([a-f0-9]{40})$')
    m = re.search(r, sha)
    if m:
        return True
    else:
        return False


def init_jerb_repo(dirname):
    """ Initializes a new jerb repo DIRNAME in the current working dir."""
    subprocess.run(['git', 'init', dirname])


def load_jerb_from_file(filepath):
    """ Loads a .jerb file and returns a Jerb object. """
    if os.path.exists(filepath):
        with open(filepath, "rb") as f:
            init_json_string = f.read()
            s = init_json_string.decode()
            j = Jerb(s)
            return j
    else:
        raise ValueError("File not found: "+filepath)


def save_jerb_to_file(jerb, filepath):
    with open(filepath, 'wb') as f:
        f.write(str(jerb))


def make_single_pack():
    """ Makes a shallow clone of this repository's last master commit only,
    then pack all it's git object files into a single .pack file """
    mref = get_master_ref()
    temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
    subprocess.call(['git', 'clone',
                     '--quiet',
                     '--no-local',
                     '--depth', '1',
                     '--branch', 'master',
                     '.', temp_repo_path])
    # Add metadata to the last commit before packing it
    md = get_repo_metadata()
    js = json.dumps(md)
    subprocess.call(['git', 'notes', 'add', '-f', '-m',
                     js, mref],
                    cwd=temp_repo_path)

    # Repack the clone into a single .pack, read out the .pack
    # then delete the cloned repo:
    subprocess.call(['git', 'repack', '-a', '-d', '--quiet'],
                    cwd=temp_repo_path)
    tmppck_dir = os.path.join(temp_repo_path, PACK_DIR)
    packs = [file for file in os.listdir(tmppck_dir) if file.endswith(".pack")]

    with open(os.path.join(tmppck_dir, packs[0]), 'rb') as f:
        pack = f.read()
    # shutil.rmtree(temp_repo_path)

    # Quit if there were more than one pack files, because we screwed up!
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
          'meta': get_repo_metadata(),
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


def ensure_metadata_exists():
    """ Ensure the orphan 'jerb_metadata' object exists in .git """
    # Get the existing metadata as a dictionary d
    if not os.path.isfile('.git/jerb_metadata'):
        init_metadata()


def init_metadata():
    h = subprocess.check_output(['git', 'hash-object', '--stdin', '-w'],
                                input='jerb_metadata'.encode())
    h = h.strip().decode()
    h = subprocess.run(['git', 'update-ref', 'jerb_metadata', h])
    d = default_metadata()
    write_metadata(d)


def write_metadata(mydict):
    ensure_metadata_exists()
    js = json.dumps(mydict)
    ref = 'jerb_metadata'
    subprocess.call(['git', 'notes', 'add', '-f', '-m', js, ref])


def edit_metadata():
    """ Initializes a new jerb repo DIRNAME in the current working dir."""
    ensure_metadata_exists()
    subprocess.run(['git', 'notes', 'edit', 'jerb_metadata'])


def get_repo_metadata():
    """ Returns the metadata in the present repo. """
    ensure_metadata_exists()
    s = subprocess.check_output(['git', 'notes', 'show', 'jerb_metadata'])
    s = s.strip().decode()
    d = json.loads(s)
    return d


def add_parent_metadata(jid):
    """ Adds the jid to the parent metadata, if any exists. """
    ensure_metadata_exists()
    d = get_repo_metadata()

    # Add the JID to the "parents" list
    if 'parents' in d:
        parents = set([x.strip() for x in d['parents'].split(',')])
        parents.add(jid)
        d['parents'] = ', '.join(parents)
    else:
        d['parents'] = jid

    write_metadata(d)


def recreate_git_notes(jid):
    """ Tries to rediscover buried notes whos refs were lost when packing. """
    cmts = subprocess.check_output(['git', 'rev-list', '--all'])
    print(cmts)


def find_only_note_object_in_index(gitdir, indexfile):
    indexpath = os.path.join(gitdir, indexfile)
    with open(indexpath, 'rb') as idxfile:
        contents = idxfile.read()
    hashes = subprocess.check_output(['git', 'show-index'],
                                     input=contents)

    commits = []
    for l in hashes.decode().splitlines():
        fields = l.split(" ")
        h = fields[1]
        t = subprocess.check_output(['git', 'cat-file', '-t', h],
                                    cwd=gitdir)
        t = t.decode().rstrip()
        commits.append(h)

    note_commit = None
    for h in commits:
        v = subprocess.check_output(['git', 'cat-file', '-p', h],
                                    cwd=gitdir)
        v = v.decode().rstrip()
        if "Notes added by 'git notes add'" in v:
            if note_commit:
                ragequit('Two note objects were found, which is impossible!')
            else:
                note_commit = h

    return note_commit


def unpack_jerb(jerb):
    """ Unpacks the jerb into the current git repo. """

    temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
    init_jerb_repo(temp_repo_path)

    # Unpack the pack file
    with open('/dev/null', 'w') as devnull:
        subprocess.run(['git', 'index-pack', '--stdin', '--keep'],
                       input=jerb.pack, stdout=devnull,
                       cwd=temp_repo_path)

    # List the hashes of all the indexed files
    tmppck_dir = os.path.join(temp_repo_path, PACK_DIR)
    idxs = [file for file in os.listdir(tmppck_dir) if file.endswith(".idx")]
    if 1 != len(idxs):
        ragequit('Error: More than one .idx file found:\n')
    note_commit = find_only_note_object_in_index(tmppck_dir, idxs[0])
    # print('Note commit is:', note_commit)

    # Destroy our temporary directory
    shutil.rmtree(temp_repo_path)

    # Finally, do the 'real thing'
    with open('/dev/null', 'w') as devnull:
        subprocess.run(['git', 'index-pack', '--stdin', '--keep'],
                       input=jerb.pack, stdout=devnull)

    subprocess.call(['git', 'merge', jerb.jid, '--quiet', '--no-edit'])
    # Carefully update the refs for notes and metadata
    subprocess.call(['git', 'update-ref',
                     'refs/notes/temp_jerb_metadata', note_commit])
    add_parent_metadata(jerb.jid)
    subprocess.call(['git', 'notes', 'merge', 'temp_jerb_metadata'])

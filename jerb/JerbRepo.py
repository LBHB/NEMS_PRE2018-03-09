import os
import binascii
import json
import shutil
import uuid
import subprocess as sub
from jerb.Jerb import Jerb

# TODO: Put git config "JERB.USER.NAME" variable somewhere?

# Constants
TMP_DIR = '/tmp/'
PACK_DIR = '.git/objects/pack/'
MASTER_REF_PATH = '.git/refs/heads/master'
JERB_METADATA_OBJ = 'jerb_metadata'


class JerbRepo():
    def __init__(self, repopath, create=False):
        """ An object that will stay in correspondance with the git/jerb
        repo that lives at repopath. """
        self.repopath = repopath
        self.reponame = os.path.basename(repopath)
        self.gitdirpath = os.path.join(self.repopath, '.git/')
        self.packdirpath = os.path.join(self.repopath, PACK_DIR)

        if create and self.git_dir_exists():
            raise ValueError("Refusing to create/overwrite existing JerbRepo")
        # Ensure that the directory has the necessary hidden files:
        if not self._git_dir_exists():
            self.init_git()
        if not self._jerb_metadata_object_exists():
            self.init_metadata()

    def _git_dir_exists(self):
        """ True when a git repo exists, false otherwise """
        return os.path.isdir(self.gitdirpath)

    def _jerb_metadata_object_exists(self):
        """ True when a git repo exists, false otherwise """
        return os.path.isfile(os.path.join(self.gitdirpath,
                                           JERB_METADATA_OBJ))

    def init_git(self):
        """ Initializes the underlying git repo at the base of a JerbRepo """
        sub.run(['git', 'init', self.reponame])

    def init_metadata(self):
        """ Creates the jerb_metadata object blob and ref. """
        ho = JERB_METADATA_OBJ
        h = sub.check_output(['git', 'hash-object', '--stdin', '-w'],
                             input=ho.encode(),
                             cwd=self.repopath)
        h = h.strip().decode()
        h = sub.run(['git', 'update-ref',
                     JERB_METADATA_OBJ, h],
                    cwd=self.repopath)
        d = self.default_metadata()
        self.set_metadata(d)

    def default_metadata(self):
        """ Try to create automatic metadata from the current git dir. """
        name = sub.check_output(['git', 'config', '--get', 'user.name'],
                                cwd=self.repopath)
        email = sub.check_output(['git', 'config', '--get', 'user.email'],
                                 cwd=self.repopath)

        branch = self.reponame
        md = {'user.name': name.strip().decode(),
              'user.email': email.strip().decode(),
              'branch': branch,
              'parents': '',
              'tags': '',
              'description': ''}
        return md

    def get_metadata(self):
        """ Returns the metadata in the present repo. """
        s = sub.check_output(['git', 'notes', 'show',
                              JERB_METADATA_OBJ],
                             cwd=self.repopath)
        s = s.strip().decode()
        d = json.loads(s)
        return d

    def set_metadata(self, metadata_dict):
        """ Write the metadata_dict to the underlying git repo's notes """
        js = json.dumps(metadata_dict)
        ref = JERB_METADATA_OBJ
        sub.call(['git', 'notes', 'add', '-f', '-m', js, ref],
                 cwd=self.repopath)

    def edit_metadata_interactively(self):
        """ Interactively edit the metadata for this JerbRepo. """
        sub.run(['git', 'notes', 'edit',
                 JERB_METADATA_OBJ],
                cwd=self.repopath)

    def add_parent(self, jid):
        """ Adds the parent jid to this repo's metadata."""
        d = self.get_metadata()
        # Add the JID to the "parents" list
        if 'parents' in d:
            parents = set([x.strip() for x in d['parents'].split(',')])
            parents.add(jid)
            d['parents'] = ', '.join(parents)
        else:
            d['parents'] = jid
        self.set_metadata(d)

    def get_master_ref(self):
        """ Returns the commit hash of the master ref. """
        mrefpath = os.path.join(self.repopath, MASTER_REF_PATH)
        with open(mrefpath, 'rb') as f:
            master_ref = f.read()
        return master_ref.decode().rstrip('\n')

    def packed_contents(self):
        """ Returns contents of .pack file containing a shallow clone
        of this repository's last master commit + the metadata note."""
        mref = self.get_master_ref()
        # TODO: Use temporary dir instead of doing this manually
        temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
        sub.call(['git', 'clone',
                         '--quiet',
                         '--no-local',
                         '--depth', '1',
                         '--branch', 'master',
                         '.', temp_repo_path])
        # Add metadata to the last commit before packing it
        # TODO: Function that adds timestamp to metadata too
        md = self.get_metadata()
        js = json.dumps(md)
        sub.call(['git', 'notes', 'add', '-f', '-m', js, mref],
                 cwd=temp_repo_path)
        # Repack the clone into an assumed single .pack, read it out,
        # then delete the temporary cloned repo
        sub.call(['git', 'repack', '-a', '-d', '--quiet'],
                 cwd=temp_repo_path)
        tmppck_dir = os.path.join(temp_repo_path, PACK_DIR)
        packs = [f for f in os.listdir(tmppck_dir) if f.endswith(".pack")]
        if 1 != len(packs):
            raise ValueError('More than one .pack file found:\n')
        with open(os.path.join(tmppck_dir, packs[0]), 'rb') as f:
            contents = f.read()
        shutil.rmtree(temp_repo_path)
        return contents

    def as_json(self):
        """ Pack and returns this repo in a JSON (string) form. """
        packfile = self.packed_contents()
        od = {'jid': self.get_master_ref,
              'meta': self.get_metadata(),
              'pack': binascii.b2a_base64(packfile).decode()}
        js = json.dumps(od, sort_keys=True)
        return js

    def as_jerb(self):
        """ Pack and returns this repo in Jerb object form. """
        js = self.as_jerb_string()
        j = Jerb(js)
        return j

    def merge_in_jerb(self, jerb_to_merge):
        """ Unpack jerb_to_merge jerb and merges it into this JerbRepo """
        temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
        new_jr = unpack_jerb(jerb_to_merge, temp_repo_path)
        note_commit = new_jr._find_note_object()
        # Merge the unpacked main commit  of the main commit
        sub.call(['git', 'merge', jerb_to_merge.jid, '--quiet', '--no-edit'])
        sub.call(['git', 'update-ref', 'refs/notes/temp_jerb_metadata',
                  note_commit])
        self.add_parent(jerb_to_merge.jid)
        sub.call(['git', 'notes', 'merge', 'temp_jerb_metadata'])
        # TODO: delete temp_jerb_metadata ref??

    def _find_note_object(self):
        """ WARNING: This function will probably throw an exception
        unless you run it on a freshly unpacked jerb. It is not intended
        to be used on anything but fresh JerbRepos, because non-fresh
        JerbRepos may have more than 1 note object. It is SLooow, also."""
        # List the hashes of all the indexed files
        objectfiles = os.listdir(self.packdirpath)
        idxs = [f for f in objectfiles if f.endswith(".idx")]
        if 1 != len(idxs):
            raise ValueError('More than one .idx file found:\n')
        idxfile = idxs[0]

        # Find the only note object in that index file
        idxpath = os.path.join(self.packdirpath, idxfile)
        with open(idxpath, 'rb') as f:
            contents = f.read()
            hashes = sub.check_output(['git', 'show-index'],
                                      input=contents,
                                      cwd=self.repopath)
        # Build a list of all repo commits
        commits = []
        for l in hashes.decode().splitlines():
            fields = l.split(" ")
            h = fields[1]
            t = sub.check_output(['git', 'cat-file', '-t', h],
                                 cwd=self.repopath)
            t = t.decode().rstrip()
            commits.append(h)

        # Filter those commits for the note commit
        note_commit = None
        for h in commits:
            v = sub.check_output(['git', 'cat-file', '-p', h],
                                 cwd=self.repopath)
            v = v.decode().rstrip()
            if "Notes added by 'git notes add'" in v:
                if note_commit:
                    raise ValueError('2 note objects found; invalid jerb!')
                else:
                    note_commit = h


# Helper function because python doesn't have multiple constructors
def unpack_jerb(jerb_to_unpack, new_jerbrepo_path):
    """ Unpacks the jerb_to_unpack Jerb object into a newly-created
    new_jerbrepo_path, and return the tuple:
    (new_jerbrepo, note_commit_hash)  """
    newrepo = JerbRepo(new_jerbrepo_path, create=True)

    # Unpack the pack file into the new repo
    binpack = binascii.a2b_base64(jerb_to_unpack.pack)
    with open('/dev/null', 'w') as devnull:
        sub.run(['git', 'index-pack', '--stdin', '--keep'],
                input=binpack,
                stdout=devnull,
                cwd=newrepo.repopath)

    return newrepo

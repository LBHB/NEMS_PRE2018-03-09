import os
import binascii
import datetime
import json
import shutil
import uuid
import subprocess as sub
from jerb.Jerb import Jerb

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

        if create and self._git_dir_exists():
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
        os.mkdir(self.repopath)
        sub.run(['git', 'init', '.'],
                cwd=self.repopath)

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
        name = sub.check_output(['git', 'config', '--get', 'jerb.user'],
                                cwd=self.repopath)
        if not name:
            raise ValueError('Please set the jerb.user variable with:',
                             'git config --global jerb.user "myusernamehere"')
        branch = self.reponame
        md = {'user': name.strip().lower().decode(),
              'branch': branch,
              'parents': [],
              'tags': [],
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

    def set_metadata_item(self, key, value):
        """ Sets the key-value pair for the metadata """
        md = self.get_metadata()
        md[key] = value
        self.set_metadata(md)

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
            parents = set(d['parents'])
            parents.add(jid)
            d['parents'] = [p for p in parents]
        else:
            d['parents'] = [jid]
        self.set_metadata(d)

    def _get_master_ref(self):
        """ Returns the commit hash of the master ref. """
        mrefpath = os.path.join(self.repopath, MASTER_REF_PATH)
        with open(mrefpath, 'rb') as f:
            master_ref = f.read()
        return master_ref.decode().rstrip('\n')

    def packed_contents(self):
        """ Returns contents of .pack file containing a shallow clone
        of this repository's last master commit + the metadata note."""
        mref = self._get_master_ref()
        # TODO: Use temporary dir instead of doing this manually
        temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
        sub.call(['git', 'clone',
                         '--quiet',
                         '--no-local',
                         '--depth', '1',
                         '--branch', 'master',
                         '.', temp_repo_path])
        # Add metadata to the last commit before packing it
        now = datetime.datetime.utcnow().replace(microsecond=0).isoformat()
        self.set_metadata_item('date', now)
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
        od = {'jid': self._get_master_ref(),
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
        # First do the 'temporary' run to find the note commit
        temp_repo_path = os.path.join(TMP_DIR, str(uuid.uuid4()))
        new_jr = JerbRepo(temp_repo_path)
        new_jr.unpack_jerb(jerb_to_merge)
        note_commit = new_jr._find_note_object()
        shutil.rmtree(temp_repo_path)
        if not note_commit:
            raise ValueError("Could not find note_commit.")
        # TODO: can I fetch from the temporary dir somehow,
        # instead of re-unpacking this thing?

        # Now actually do the merge
        self.unpack_jerb(jerb_to_merge)
        sub.call(['git', 'merge', jerb_to_merge.jid, '--quiet', '--no-edit'],
                 cwd=self.repopath)
        sub.call(['git', 'update-ref', 'refs/notes/temp_jerb_metadata',
                  note_commit],
                 cwd=self.repopath)
        self.add_parent(jerb_to_merge.jid)
        sub.call(['git', 'notes', 'merge', 'temp_jerb_metadata'],
                 cwd=self.repopath)

    def _find_note_object(self):
        """ WARNING: This function will probably throw an exception
        unless you run it on a freshly unpacked jerb. It is not intended
        to be used on anything but JerbRepos just made from a Jerb, because
        in general JerbRepos may have >1 note objects. It is SLooow, too!"""
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
        return note_commit

    def unpack_jerb(self, jerb_to_unpack):
        """ Unpacks the jerb_to_unpack Jerb object into this repo."""
        binpack = binascii.a2b_base64(jerb_to_unpack.pack)
        with open('/dev/null', 'w') as devnull:
            sub.run(['git', 'index-pack', '--stdin', '--keep'],
                    input=binpack,
                    stdout=devnull,
                    cwd=self.repopath)

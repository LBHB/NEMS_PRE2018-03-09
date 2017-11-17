""" jerbstore : An API for storing jerbs centrally
Presently, by default it stores them in a directory. """

import os
import botocore
import boto3
import io

from flask import abort, request, Response
from flask_restful import Resource

from jerb.Jerb import Jerb, valid_SHA1_string
from jerb.JerbCentral import JerbCentral

S3_JERBS_BUCKET = 'jerbs'


def ensure_valid_jid(jid):
    if not valid_SHA1_string(jid):
        abort(400, 'invalid SHA1 string:' + jid)


def jerb_not_found():
    # TODO: Southpark reference
    abort(404, "jerb not found")


class LocalJerbStore(Resource):
    """ A class representing a store of local jerb files."""
    def __init__(self, **kwargs):
        self.LOCAL_JERB_DIR = kwargs['local_jerb_dir']

    def jerb_path(self, jid):
        """ Returns the path for a local JID."""
        p = os.path.join(self.LOCAL_JERB_DIR, jid[0:2], jid[2:])
        return p

    def jerb_exists(self, jid):
        """ Predicate. Returns True when the JID exists locally. """
        p = self.jerb_path(jid)
        return os.path.exists(p)

    def get(self, jid):
        """ Returns the jerb found at JID, if it exists. """
        ensure_valid_jid(jid)
        if not self.jerb_exists(jid):
            jerb_not_found()
        p = self.jerb_path(jid)
        with open(p, "rb") as f:
            d = f.read()
        return Response(d, status=200, mimetype='application/json')

    def put(self, jid):
        """ Idempotent. Returns 201 if the jerb was created, or
        return 200 if it exists already in this jerbstore."""
        ensure_valid_jid(jid)
        # TODO: Ensure request size is within limits
        j = Jerb(request.data.decode())
        if not jid == j.jid:
            abort(400, 'JID does not match argument')
        if any(j.errors()):
            abort(400, 'jerb contains errors')
        if not self.jerb_exists(jid):
            p = self.jerb_path(j.jid)
            d = os.path.dirname(p)
            if not os.path.isdir(d):
                os.makedirs(d)
            with open(p, 'wb') as f:
                f.write(request.data)
            return Response(status=201)
        else:
            return Response(status=200)

    def delete(self, jid):
        ensure_valid_jid(jid)
        # TODO
        jerb_not_found()


class CentralJerbStore(Resource):
    """ A class representing a git repo that stores jerb files."""
    def __init__(self, **kwargs):
        self.jc = JerbCentral(repopath=kwargs['jerb_central_repo_dir'])

    def jerb_exists(self, jid):
        """ Predicate. Returns True when the JID exists locally. """
        # TODO
        return False

    def get(self, jid):
        """ Returns the jerb found at JID, if it exists. """
        ensure_valid_jid(jid)
        # if not self.jerb_exists(jid): # TODO
        #    jerb_not_found()
        d = self.jc.emit_jerb(jid)
        return Response(d, status=200, mimetype='application/json')

    def put(self, jid):
        """ Idempotent. Returns 201 if the jerb was created, or
        return 200 if it exists already in this jerbstore."""
        ensure_valid_jid(jid)
        # TODO: Ensure request size is within limits
        j = Jerb(request.data.decode())
        if not jid == j.jid:
            abort(400, 'JID does not match argument')
        if any(j.errors()):
            abort(400, 'jerb contains errors')
        if not self.jerb_exists(jid):
            self.jc.absorb_jerb(j)
            return Response(status=201)
        else:
            return Response(status=200)

    def delete(self, jid):
        ensure_valid_jid(jid)
        # TODO
        jerb_not_found()


class S3JerbStore(Resource):
    """ A class representing a store of jerb files in S3."""
    def __init__(self, **kwargs):
        self.s3 = boto3.client('s3')
        # Create the jerbs bucket:
        try:
            self.s3.head_bucket(Bucket=S3_JERBS_BUCKET)
        except botocore.exceptions.ClientError as e:
            # If a client error is thrown, then check that it was a 404 error.
            # If it was a 404 error, then the bucket does not exist. Create it.
            error_code = int(e.response['Error']['Code'])
            if error_code == 404:
                self.jb = self.s3.create_bucket(
                    Bucket=S3_JERBS_BUCKET,
                    ACL='public-read',
                    CreateBucketConfiguration={'LocationConstraint':
                                               'us-west-2'})

    def jerb_exists(self, jid):
        """ Wow, either Boto3 design is bollocks or I am missing the obvious
        way of testing existance. Control flow exception trapping logic is
        shameful! """
        try:
            self.s3.head_object(Bucket=S3_JERBS_BUCKET,
                                Key=jid)
        except botocore.errorfactory.ClientError:
            return False
        return True

    def get(self, jid):
        """ Returns the jerb found at JID, if it exists. """
        ensure_valid_jid(jid)
        if not self.jerb_exists(jid):
            jerb_not_found()
        jf = io.BytesIO()
        self.s3.download_fileobj(S3_JERBS_BUCKET, jid, jf)
        d = jf.getvalue().decode()
        jf.close()
        return Response(d, status=200, mimetype='application/json')

    def put(self, jid):
        """ Idempotent. Returns 201 if the jerb was created, or
        return 200 if it exists already in this jerbstore."""
        ensure_valid_jid(jid)
        # TODO: Ensure request size is within limits
        j = Jerb(request.data.decode())
        if not jid == j.jid:
            abort(400, 'JID does not match argument')
        if any(j.errors()):
            abort(400, 'jerb contains errors')
        if not self.jerb_exists(jid):
            jf = io.BytesIO(str(j).encode())
            self.s3.upload_fileobj(jf, S3_JERBS_BUCKET, jid)
            jf.close()
            return Response(status=201)
        else:
            return Response(status=200)

    def delete(self, jid):
        ensure_valid_jid(jid)
        if self.jerb_exists(jid):
            self.s3.delete_object(Bucket=S3_JERBS_BUCKET,
                                  Key=jid)

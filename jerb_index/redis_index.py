import json
import redis
import uuid


# Redis Schema Cheat Sheet:

# THIS KEY STRING (->mapsto->)  THESE OBJECTS
# ---------------------------------------------
#  jid:JID           -> JSON string           (Forward index)
#  idx:field=value   -> Set of SHA256s        (Reverse indices)
#  prop:field        -> Set of values seen so far for that prop
#  br:user/branch    -> JID          (Essentially git refs)
#  brt:user/branch   -> JID          (Essentially git refs)

def redis_connect(credentials):
    """ Return a redis connection specified by the given credentials """
    required_creds = ['REDIS_HOST',
                      'REDIS_PORT',
                      'REDIS_PASS']
    if not all(c in credentials for c in required_creds):
        raise ValueError('Required Redis credentials not all provided: '
                         + str(required_creds))
    else:
        r = redis.Redis(host=credentials['REDIS_HOST'],
                        port=credentials['REDIS_PORT'],
                        password=credentials['REDIS_PASS'])
        return r


def index_jerb(r, jerb):
    """ Index a JERB so that it may be searched. Please do
    any consistency and error checking before you call this. """
    # TODO : Only index the jerb if it is not already indexed
    # TODO : start transaction
    forward_index(r, jerb)
    reverse_index(r, jerb)
    set_head_if_newer(r, jerb)


def deindex_jerb(r, jerb):
    """ Inverse operation of index_jerb """
    delete_forward_index(r, jerb)
    delete_reverse_index(r, jerb)


###############################################################################
# Forward lookups


def forward_index(r, jerb):
    """ Create forward index from a JID to a Jerb JSON."""
    k = 'jid:' + jerb.jid
    v = json.dumps(jerb.meta)
    r.set(k, v)


def delete_forward_index(r, jerb):
    """ Inverse operation of forward_index."""
    k = 'jid:' + jerb.jid
    r.delete(k)


def lookup_jid(r, jid):
    """ Forward lookup. Returns a string of what was stored at JID. """
    jrb = r.get('jid:' + jid)
    return jrb.decode()

###############################################################################
# User/Branch Lookups (Replaces git refs)


def set_head_if_newer(r, jerb):
    """ Sets the jerb as the branch head, if it is newer than HEAD, or
    if that user/branch does not already exist. """
    date = get_head_date(r, jerb.meta['user'], jerb.meta['branch'])
    if not date or (jerb.meta['date'] > date):
        # TODO: Avoid race condition that exists here!
        set_head(r, jerb)


def set_head(r, jerb):
    """ Sets the jerb as the branch head. """
    jid = jerb.jid
    user = jerb.meta['user']
    branch = jerb.meta['branch']
    date = jerb.meta['date']
    if not (user and branch and jid and date):
        raise ValueError('JID, user, date and branch are not all defined!')
    r.set('br:' + user + '/' + branch, jid)
    r.set('brt:' + user + '/' + branch, date)


def get_head(r, user, branch):
    """ Gets the JID of the user/branch."""
    if not (user and branch):
        raise ValueError('User and Branch are not defined!')
    v = r.get('br:' + user + '/' + branch)
    if v:
        return v.decode()
    else:
        return None


def get_head_date(r, user, branch):
    """ Gets the timestamp of the user/branch."""
    if not (user and branch):
        raise ValueError('User and Branch are not defined!')
    v = r.get('brt:' + user + '/' + branch)
    if v:
        return v.decode()
    else:
        return None

###############################################################################
# Reverse Lookups


def reverse_index(r, jerb):
    """ Create reverse indexes for a Jerb, so that you can search
    by a property to find the JIDs that have that property. """
    # TODO: This is a potential security vulnerability, because we don't
    # know what could come out of that property dictionary without
    # sanitizing it. Strings with ":" in them could break something?
    # TODO: Write sanitizing function? Or use redis to treat it as
    # just an uninterpretable bytestring?
    for k, v in jerb.meta.items():
        if (k and v):
            if type(v) is str:
                r.sadd('idx:'+k+'='+v, jerb.jid)
                r.sadd('prop:'+k, v)
            elif type(v) is list:
                for vv in v:
                    r.sadd('idx:'+k+'='+vv, jerb.jid)
                    r.sadd('prop:'+k, vv)


def delete_reverse_index(r, jerb):
    """ Inverse operation of reverse_index. """
    # TODO: This is potentially O(N) because you have to traverse set?
    # TODO: Use transactions here
    for k, v in jerb.meta.items():
        r.srem('idx:'+k+'='+v, jerb.jid)
        cnt = r.scard('idx:'+k+'='+v)
        if 0 >= cnt:
            r.srem('prop:'+k, v)


def lookup_prop(r, prop, val):
    """ Reverse lookup. Returns list of all JIDs that have the given metadata
    prop and val defined. """
    jids = r.smembers('idx:'+prop+'='+val)
    return [j.decode() for j in jids]


##############################################################################
# For browsing the tree


def browse_prop(r, prop):
    """ Return a list of all the values found for a given metadata property."""
    vals = r.smembers('prop:'+prop)
    return [v.decode() for v in vals]


def browse_prop_with_counts(r, prop):
    """ Return a dict of values and counts found for a given metadata prop."""
    vals = r.smembers('prop:'+prop)
    ret = {}
    for v in vals:
        v = v.decode()
        ret[v] = r.scard('idx:'+prop+'='+v)
        # TODO: If you ever got back a count of 0, do r.srem('prop:'+k, v)
    return ret


##############################################################################
# For finding JIDs


def select_jids_where(r, query):
    """ Query is a dict that maps strings to values or lists.
    Dicts define "AND" expressions: {k1=D,k2=E } means "k1=D and k2=E"
    Lists define "OR" expressions: {k=[A,B,C]} means "k=A or k=B or k=C"

    Example translation of SQL to this type of query object:
    # SELECT jid WHERE key1='this' OR key1='that' AND key2='bar'
    {'key1': ['this', 'that'], 'key2': 'bar'}

    Sorry, you cannot presently query using OR statements at the top level:
    # SELECT jid WHERE key1='this' OR key2='bar'
    ( Not yet implemented )
    """
    tmpid = uuid.uuid4()
    first_time = True
    for k, v in query.items():
        if type(v) is list:
            # List values indicate OR
            for w in v:
                r.sunionstore(tmpid, tmpid, 'idx:'+k+'='+w)
        elif type(v) is str:
            # String values indicate AND
            if first_time:
                r.sunionstore(tmpid, tmpid, 'idx:'+k+'='+v)
            else:
                r.sinterstore(tmpid, tmpid, 'idx:'+k+'='+v)
        else:
            # Anything else is unacceptable
            raise ValueError("The query spec was violated.")
        first_time = False

    ret = [v.decode() for v in r.smembers(tmpid)]
    r.delete(tmpid)
    return ret

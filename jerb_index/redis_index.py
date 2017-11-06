import json
import redis


# Redis Schema Cheat Sheet:

# THIS KEY STRING (->mapsto->)  THESE OBJECTS
# ---------------------------------------------
#  jid:JID           -> JSON string           (Forward index)
#  idx:field=value   -> Set of SHA256s        (Reverse indices)
#  prop:field        -> Set of values seen so far for that prop


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
            r.sadd('idx:'+k+'='+v, jerb.jid)
            r.sadd('prop:'+k, v)


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

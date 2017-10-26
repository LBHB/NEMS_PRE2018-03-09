import redis


# Redis Schema Cheat Sheet:

# THIS KEY STRING (mapsto)  THESE OBJECTS
# ---------------------------------------------
#  jid:JID          -> JSON string            (Forward index)
#  idx:prop=value   -> List of SHA256s        (Reverse index)
#  prop:prop        -> Set of values seen so far for that prop
#  prop:value:cnt   -> Count of times that value was seen


def redis_connect(credentials):
    """ Return a redis connection specified by the given credentials """
    required_creds = ['REDIS_HOST',
                      'REDIS_PORT',
                      'REDIS_PASS']
    if not all(c in credentials for c in required_creds):
        raise ValueError('Required Redis credentials not all provided: '
                         + required_creds)
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


def forward_index(r, jerb):
    """ Create forward index from a JID to a Jerb JSON."""
    k = "jid:" + jerb.jid
    v = jerb.as_json()
    r.set(k, v)


def delete_forward_index(r, jerb):
    """ Create forward index from a JID to a Jerb JSON."""
    k = jerb.jid
    v = jerb.as_json()
    r.set(k, v)


def reverse_index(r, jerb):
    """ Create reverse indexes for a Jerb, so that you can search
    by a property to find the JIDs that have that property. """
    # TODO: This is a potential security vulnerability, because we don't
    # know what could come out of that property dictionary without
    # sanitizing it. Strings with ":" in them could break something?
    # TODO: Write sanitizing function? Or use redis to treat it as
    # just an uninterpretable bytestring?
    jid = jerb.jid
    for k, v in jerb.props:
        r.lpush("idx:"+k+"="+v, jid)
        r.sadd("prop:"+k, v)
        r.incr("prop:"+k+"="+v+":cnt", 1)  # Keep track of occurrence count


def delete_reverse_index(r, jerb):
    """ Inverse operation of reverse_index. """
    jid = jerb.jid
    # TODO: This is potentially O(N) because you have to traverse list
    # TODO: Should we switch to ZLISTs?
    for k, v in jerb.index_props():
        r.lrem("idx:"+k+"="+v, jid)
        cnt = r.decr("prop:"+k+"="+v+":cnt")
        if 0 <= cnt:
            r.srem("prop:"+k, v)


def lookup_jid(r, jid):
    """ Forward lookup. Returns the JSON stored at JID. """
    jrb = r.get("jid:", jid)
    return jrb


def lookup_prop(r, prop, val, startat=0, limit=100):
    """ Reverse lookup. Returns list of up to 1000 the JIDs which have
    the given prop and val defined. Optional arguments STARTAT and LIMIT are
    for paging if you expect to get a very long list of JIDS back. """
    jids = r.lget("idx:" + prop + "=" + val, startat, limit)
    return jids


def browse_prop(r, prop, startat=0, limit=100):
    """ Return a list of all the values found for a given property. """
    vals = r.smembers("prop:" + prop)
    return vals

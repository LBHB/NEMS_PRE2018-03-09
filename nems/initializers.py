from nems.utils import split_keywords
from nems import keywords


def from_keywords(data, keyword_string, registry=keywords.defaults):
    '''
    Returns a modelspec sized to match the dimensionality of nems.Recording
    DATA and with the structure defined by keywords in KEYWORD_STRING.
    '''
    keywords = split_keywords(keyword_string)

    # Lookup the modelspec fragments in the registry
    modelspec = [registry[kw] for kw in keywords]

    # TODO: Actually use data and resize modelspec as necessary!
    #       OTOH: I kind of hate having a single keyword map to
    #       a variety of objects with different sizes.

    return modelspec

from nems.utils import split_keywords
from nems import keywords


def from_keywords(keyword_string, registry=keywords.defaults):
    '''
    Returns a modelspec created by splitting keyword_string on underscores
    and replacing each keyword with what is found in the nems.keywords.defaults
    registry. You may provide your own keyword registry using the
    registry={...} argument.
    '''
    keywords = split_keywords(keyword_string)

    # Lookup the modelspec fragments in the registry
    modelspec = []
    for kw in keywords:
        d = registry[kw]
        d['id'] = kw
        modelspec.append(d)

    return modelspec

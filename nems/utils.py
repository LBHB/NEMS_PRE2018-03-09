def split_keywords(keyword_string):
    '''
    Return a list of keywords resulting from splitting keyword_string.
    '''
    return keyword_string.split('_')


def split_to_api_and_fn(mystring):
    '''
    Returns (api, fn_name) given a string that would be used to import
    a function from a package.
    '''
    matches = mystring.split(sep='.')
    api = '.'.join(matches[:-1])
    fn_name = matches[-1]
    return api, fn_name

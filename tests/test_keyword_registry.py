import re
from functools import partial
import pytest


def shim_function(stack, name, groups=None):
    '''
    This is a shim that substitutes for the actual keyword function that
    modifies the stack. It simply takes the stack (a list for testing purposes)
    and adds the name of the keyword and any arguments parsed from the regex.
    The name and parsed arguments will be compared against the expected results.
    '''
    stack.append((name, groups))


def shim_parser(name, groups=None):
    return partial(shim_function, name=name, groups=groups)


def raise_error(*args, **kwargs):
    '''
    A shim that raises an error when it's called with the stack. This is used to
    ensure that if a keyword function doesn't like the arguments passed to it,
    the registry will continue to search through its list of possible matches.
    '''
    raise ValueError('Invalid value')


@pytest.fixture
def registry():
    from nems.keyword.registry import KeywordRegistry
    registry = KeywordRegistry()
    registry.update({
        re.compile(r'^wc(\w)??(\d{2})$'): partial(shim_parser, 'wc'),
        re.compile(r'^fir(\d{2})(\w)??$'): partial(shim_parser, 'fir'),
        'stp1pc': partial(shim_function, name='stp'),
        'fit01': partial(shim_function, name='fitter'),
        re.compile(r'^zz(\d{2})$'): raise_error,
    })

    # Add this after the update so that we can ensure that this is discovered
    # after the pattern regex for `zz` is found to match, but fails (due to the
    # `raise_error`).
    registry['zz01'] = partial(shim_function, name='zz')
    return registry


def test_key_parser(registry):
    '''
    This simply asserts that the keyword regex lookup system works and that if
    an apparent match fails it continues the search (e.g., zz01 should match the
    regex, but the associated keyword function raises an error so it should
    continue on to match with the string zz01), it continues the search.
    '''
    tests = {
        'wc01': ('wc', (None, '01')),
        'wcg01': ('wc', ('g', '01')),
        'stp1pc': ('stp', None),
        'zz01': ('zz', None),
    }
    for keyword, expected in tests.items():
        function = registry[keyword]
        stack = []
        function(stack)
        assert stack[0] == expected

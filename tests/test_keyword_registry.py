import re
from functools import partial
import pytest


def shim_function(stack, name, groups=None):
    stack.append((name, groups))


def shim_parser(name, groups=None):
    return partial(shim_function, name=name, groups=groups)


@pytest.fixture
def registry():
    from nems.keyword.registry import KeywordRegistry
    registry = KeywordRegistry()
    registry.update({
        re.compile(r'wc(\w)??(\d{2})'): partial(shim_parser, 'wc'),
        re.compile(r'fir(\d{2})(\w)??'): partial(shim_parser, 'fir'),
        'stp1pc': partial(shim_function, name='stp'),
        'fit01': partial(shim_function, name='fitter'),
    })
    return registry


def test_key_parser(registry):
    tests = {
        'wc01': ('wc', (None, '01')),
        'wcg01': ('wc', ('g', '01')),
        'stp1pc': ('stp', None),
    }
    for keyword, expected in tests.items():
        function = registry[keyword]
        stack = []
        function(stack)
        assert stack[0] == expected


def test_model_name_parser(registry):
    model_name = 'wc01_stp1pc_fit01'
    expected = [('wc', (None, '01')),
                ('stp', None),
                ('fitter', None),
                ]
    stack = []
    for keyword in model_name.split('_'):
        registry[keyword](stack)
    assert stack == expected

from functools import partial

import nems.utilities as ut
from .registry import keyword_registry


def nested(stack, folds):
    """
    Keyword for k-fold nested crossvalidation. Uses 5% validation chunks.  This
    must be the last keyworod in the modelname string and cannot be included
    twice.
    """
    ut.utils.nest_helper(stack, nests=folds)


keyword_registry.update({
    'nested20': partial(nested, folds=20),
    'nested10': partial(nested, folds=10),
    'nested5': partial(nested, folds=5),
    'nested2': partial(nested, folds=2),
})

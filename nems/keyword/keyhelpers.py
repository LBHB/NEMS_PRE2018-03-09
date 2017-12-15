#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keyword helper functions

Created on Fri Aug 11 10:39:17 2017

@author: shofer
"""

import nems.utilities as ut


# Nested Crossval
###############################################################################

def nested20(stack):
    """
    Keyword for 20-fold nested crossvalidation. Uses 5% validation chunks.
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    ut.utils.nest_helper(stack, nests=20)


def nested10(stack):
    """
    Keyword for 10-fold nested crossvalidation. Uses 10% validation chunks.
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    ut.utils.nest_helper(stack, nests=10)


def nested5(stack):
    """
    Keyword for 5-fold nested crossvalidation. Uses 20% validation chunks.
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    ut.utils.nest_helper(stack, nests=5)


def nested2(stack):
    """
    Keyword for 2-fold nested crossvalidation. Uses 50% validation chunks.
    MUST be last keyowrd in modelname string. DO NOT include twice.
    """
    ut.utils.nest_helper(stack, nests=2)

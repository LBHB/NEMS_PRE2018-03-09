#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Est/val keywords

Est/val is usually incorporated into most loader keywords, but these work if
an est/val module is not included in the loader keyword

Created on Fri Aug 11 10:36:16 2017

@author: shofer
"""

import nems.modules as nm

__all__ = ['ev', 'xval05', 'xval10']


def ev(stack):
    """
    Breaks the data into estimation and validation datasets based on the number
    of trials of each stimulus.
    """
    stack.append(nm.est_val.standard, valfrac=0.05)


def xval10(stack):
    """
    Breaks the data into estimation and validation datasets by placing 90% of the
    trials/stimuli into the estimation set and 10% into the validation set.
    """
    stack.append(nm.est_val.crossval, valfrac=0.1)


def xval05(stack):
    """
    Breaks the data into estimation and validation datasets by placing 95% of the
    trials/stimuli into the estimation set and 5% into the validation set.
    """
    stack.append(nm.est_val.crossval, valfrac=0.05)

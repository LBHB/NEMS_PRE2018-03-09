import pytest

import numpy as np

from nems.epoch import (epoch_union, epoch_difference, epoch_intersection,
                        epoch_contains, adjust_epoch_bounds)

@pytest.fixture()
def epoch_a():
    return np.array([
        [  0,  50],
        [ 60,  70],
        [ 75,  76],
        [ 77,  77],
        [ 85, 100],
     ])


@pytest.fixture()
def epoch_b():
    return np.array([
        [ 55,  70],
        [ 75,  76],
        [ 90,  95],
        [110, 120],
     ])


def test_intersection(epoch_a, epoch_b):
    expected = np.array([
        [ 60,  70],
        [ 75,  76],
        [ 90,  95],
    ])
    result = epoch_intersection(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_union(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 55,  70],
        [ 75,  76],
        [ 77,  77],
        [ 85, 100],
        [110, 120],
    ])
    result = epoch_union(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_difference(epoch_a, epoch_b):
    expected = np.array([
        [  0,  50],
        [ 77,  77],
        [ 85,  90],
        [ 95, 100],
    ])
    result = epoch_difference(epoch_a, epoch_b)
    assert np.all(result == expected)


def test_contains(epoch_a, epoch_b):
    expected_any = np.array([
        False,
        True,
        True,
        False,
        True,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'any')
    assert np.all(actual == expected_any)

    expected_start = np.array([
        False,
        False,
        True,
        False,
        True,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'start')
    assert np.all(actual == expected_start)

    expected_end = np.array([
        False,
        False,
        False,
        False,
        True,
    ])
    actual = epoch_contains(epoch_a, epoch_b, 'end')
    assert np.all(actual == expected_end)

    expected_both = expected_start & expected_end
    actual = epoch_contains(epoch_a, epoch_b, 'both')
    assert np.all(actual == expected_both)


def test_adjust_epoch_bounds(epoch_a):
    expected = epoch_a + np.array([-1, 0])
    actual = adjust_epoch_bounds(epoch_a, -1)
    assert np.all(actual == expected)

    expected = epoch_a + np.array([0, 5])
    actual = adjust_epoch_bounds(epoch_a, 0, 5)
    assert np.all(actual == expected)

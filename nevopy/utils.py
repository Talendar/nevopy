"""
Utility functions.
"""

import numpy as np


def chance(p):
    """ Randomly returns True or False. The parameter specifies the chance of returning True."""
    return np.random.uniform(low=0, high=1) < p


def align_lists(lists, getkey=None, placeholder=None):
    """ Aligns the given lists based on their common values. Repeated entries within a single list are discarded.

    Example of use:
        >>> align_lists(([1, 2, 3, 6], [1, 3, 4, 5]))
        [[1, 2, 3, None, None, 6], [1, None, 3, 4, 5, None]]

    :param lists: list containing the lists to be alligned.
    :param getkey: function used to retrieve the keys of the lists entries.
    :param placeholder: value to be placed as a placeholder on the aligned lists.
    :return: list containing the aligned lists (the original order is preserved).
    """
    union = set()
    for l in lists:
        union = union | set(l)
    union = sorted(union, key=getkey)

    result = []
    for l in lists:
        result.append([n if n in l else placeholder for n in union])
    return result

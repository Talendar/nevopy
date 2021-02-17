# MIT License
#
# Copyright (c) 2020 Gabriel Nogueira (Talendar)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================

""" This module implements useful utility functions.
"""

import os
import pickle
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import (Any, Callable, Iterable, List, Optional, Set, TypeVar,
                    Sequence, Tuple, Generic)

import click
import numpy as np
from click import style

#: `TypeVar` indicating an undefined type
_T = TypeVar("_T")


class Comparable(metaclass=ABCMeta):
    """ Indication of a "comparable" type. """
    @abstractmethod
    def __lt__(self, other: Any) -> bool:
        pass


class MutableWrapper(Generic[_T]):
    """ Simple class for wrapping immutable objects so they can be passed by
    reference to a callable.
    """
    def __init__(self, value: _T) -> None:
        self.value = value


def chance(p: float) -> bool:
    """ Randomly returns `True` or `False`.

    Args:
        p (float): Float between 0 and 1. Specifies the chance of the function
            returning True.

    Returns:
        A randomly chosen boolean value (`True` or `False`).
    """
    return np.random.uniform(low=0, high=1) < p


def pickle_save(obj: Any, abs_path: str) -> None:
    """ Saves the given object to the given absolute path.

    Simple wrapper around the `pickle` package.

    Args:
        obj (Any): Object to be saved.
        abs_path (str): Absolute path of the saving file. If the given path
            doesn't end with the suffix ".pkl", it will be automatically
            added.
    """
    p = Path(abs_path)
    if not p.suffixes:
        p = Path(str(abs_path) + ".pkl")
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(str(p), "wb") as out_file:
        pickle.dump(obj, out_file, pickle.HIGHEST_PROTOCOL)


def pickle_load(abs_path: str) -> Any:
    """ Loads an object from the given absolute path.

    Simple wrapper around the `pickle` package.

    Args:
        abs_path (str): Absolute path of the saved ".pkl" file. If the given
            path doesn't end with the suffix ".pkl", it will be automatically
            added.

    Returns:
        The loaded object.
    """
    p = Path(abs_path)
    if not p.suffixes:
        p = Path(str(abs_path) + ".pkl")

    with open(p, "rb") as in_file:
        return pickle.load(in_file)


def make_xor_data(num_variables: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """ Builds data using the `XOR` logic function.

    The generated inputs are all the possible combinations of input values with
    the specified number of variables. Each variable is a bit (0 or 1). The
    generated outputs are the results (a single bit each) of the `XOR` function
    applied to all the inputs.

    Example:

        >>> xor_in, xor_out = make_xor_data(num_variables=2)
        >>> for x, y in zip(xor_in, xor_out):
        ...     print(f"{x} -> {y}")
        ...
        [0 0] -> 0
        [0 1] -> 1
        [1 0] -> 1
        [1 1] -> 0

    Args:
        num_variables (int): Number of input variables for the `XOR` logic
            function.

    Returns:
        A tuple with two numpy arrays. The first array contains the input
        values, and the second array contains the output of the `XOR` function.
    """
    assert num_variables >= 2, "The XOR function needs at least 2 variables!"

    xor_inputs, xor_outputs = [], []
    for num in range(2 ** num_variables):
        binary = bin(num)[2:].zfill(num_variables)
        xin = [int(binary[0])]
        xout = int(binary[0])

        for bit in binary[1:]:
            xin.append(int(bit))
            xout ^= int(bit)

        xor_inputs.append(np.array(xin))
        xor_outputs.append(np.array(xout))

    return np.array(xor_inputs), np.array(xor_outputs)


def align_lists(lists: Iterable[List[_T]],
                getkey: Optional[Callable[[_T], Comparable]] = None,
                placeholder: Optional[Any] = None) -> List[List[_T]]:
    """ Aligns the given lists based on their common values.

    Repeated entries within a single list are discarded.

    Example:
        >>> align_lists(([1, 2, 3, 6], [1, 3, 4, 5]))
        [[1, 2, 3, None, None, 6], [1, None, 3, 4, 5, None]]

    Args:
        lists (Iterable[List[_T]]): Iterable that yields lists containing the
            objects to be aligned.
        getkey (Optional[Callable[[_T], Comparable]]): Optional function to be
            passed to :py:func:`sorted()` to retrieve comparable keys from the
            objects to be aligned.
        placeholder (Optional[Any]): Value to be used as a placeholder when an
            item doesn't match with any other (see the example above, where
            `None` is the placeholder).

    Returns:
        A list containing the aligned lists. The items in the aligning lists
        will be sorted in ascending order.
    """
    union = set()  # type: Set[_T]
    for lst in lists:
        union = union | set(lst)
    values = sorted(union, key=getkey)

    result = []
    for lst in lists:
        result.append([n if n in lst else placeholder for n in values])
    return result


def min_max_norm(values: Iterable) -> np.array:
    """ Applies min-max normalization to the given values. """
    a = np.array(values)
    a_min, a_max = np.min(a), np.max(a)
    return (a - a_min) / (a_max - a_min)


def rank_prob_dist(size: int,
                   coefficient: float,
                   min_prob: float = 1e-9) -> np.ndarray:
    """ Calculates a probability distribution that associates a probability to
    each position of a rank with the given size.

    Args:
        size (int): Size of the rank (and of the probability distribution).
        coefficient (float): This constant (let's call it `c`) can be
            interpreted as follows: the position `p` of the rank is assigned a
            probability that is `c` times higher than the position `p + 1` of
            the rank. If c = 2, here is an example of a probability distribution
            generated by this function: `[0.5, 0.25, 0.125, 0.0675, ...]`.
        min_prob (float): Probabilities with a value lower than the value passed
            to this parameter will be converted to 0. This prevents the
            occurrence of an arithmetic underflow.

    Returns:
        A numpy array with the probability distribution. The value in the index
        `i` of the array represents the probability of the position `i` of the
        rank.
    """
    prob = np.zeros(size)
    prob[0] = 1 - 1 / coefficient
    for i in range(1, size):
        p = prob[i - 1] / coefficient
        if p < min_prob:
            break
        prob[i] = p
    return prob


def make_table_row(name: str,
                   current: float,
                   past: float,
                   abs_format: str = ".2E",
                   inc_format: str = "+0.2E",
                   pc_format: str = "+0.2%",
                   show_inc_pc: bool = True,
                   colors: bool = True,
                   positive_color: str = "green",
                   negative_color: str = "red",
                   neutral_color: str = "white") -> List[str]:
    """ Makes a row for a `columnar` table.

    Information in the row: name of the attribute; current value of the
    attribute; past value of the attribute; how much the attribute increased
    (absolute and percentage).
    """
    inc = f"{current - past:{inc_format}}"
    if colors:
        inc = style(inc, fg=(positive_color if current > past
                             else negative_color if current < past
                             else neutral_color))

    inc_pc = "-"
    if show_inc_pc:
        pc = float("inf") if past == 0 else (current - past) / past
        pc = abs(pc) * (1, -1)[current < past]
        inc_pc = f"{pc:{pc_format}}"
        if colors:
            inc_pc = style(inc_pc, fg=(positive_color if current > past
                                       else negative_color if current < past
                                       else neutral_color))

    return [name,
            f"{current:{abs_format}}", f"{past:{abs_format}}",
            inc, inc_pc]


def is_jupyter_notebook() -> bool:
    """ Checks whether the program is running on a jupyter notebook.

    Warning:
        This function is not guaranteed to work! It simply checks if
        :py:func:`IPython.get_ipython` returns `None`.

    Returns:
        `True` if the program is running on a jupyter notebook and `False`
        otherwise.
    """
    try:
        # pylint: disable=import-outside-toplevel
        from IPython import get_ipython
        if get_ipython() is None:
            return False
    except ModuleNotFoundError:
        return False
    return True


def clear_output() -> None:
    """ Clears the output.

    Should work on Windows and Linux terminals and on Jupyter notebooks. On
    PyCharm, it simply prints a bunch of new lines.
    """
    if "PYCHARM_HOSTED" in os.environ:
        print("\n" * 15)
    elif is_jupyter_notebook():
        # pylint: disable=import-outside-toplevel
        from IPython.display import clear_output as jupyter_clear
        jupyter_clear(wait=True)
    else:
        click.clear()


def round_proportional_distribution(to_distribute: int,
                                    values: Sequence[float]) -> List[int]:
    """ Given an integer `A` and a sequence `S` of arbitrary size `k`, this
    function divides `A` into `k` integers. The proportion that the `i-th`
    integer represents of `A` is approximately equal to the proportion that
    `S[i]` represents of `sum(S[i])`.

    Example:

        >>> round_proportional_distribution(100, [1.22, 2.78, 0.26, 5.74])
        [12, 28, 3, 57]

    Args:
        to_distribute (int): Integer to be distributed.
        values (Sequence[float]): Values that will serve as a reference.

    Returns:
        A list with the same size as ``values`` containing integers that sum to
        ``to_distribute``.
    """
    total = np.sum(values)
    percentages = [v / total for v in values]
    dist = [int(to_distribute * pc) for pc in percentages]

    remainder = to_distribute - np.sum(dist)
    if remainder > 0:
        decimals = sorted([(i, 100 * pc - int(100 * pc))
                           for i, pc in enumerate(percentages)],
                          reverse=True, key=lambda n: n[1])
        for _ in range(remainder):
            i, _ = decimals.pop(0)
            dist[i] += 1

    assert to_distribute == np.sum(dist)
    return dist

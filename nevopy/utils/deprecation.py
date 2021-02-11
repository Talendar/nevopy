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

""" Implements a decorator that can be used to mark functions, methods or
classes as being deprecated.
"""

import functools
import inspect
import logging
from typing import Any, Dict, Optional


_logger = logging.getLogger(__name__)

#: Allows deprecation warnings to be silenced.
_PRINT_DEPRECATION_WARNINGS = True

#: Counts how many times deprecated items have been called. If a given item is
#:  is not present, it haven't been called it.
_PRINTED_WARNINGS_COUNT = {}  # type: Dict[Any, int]


def _get_call_location() -> str:
    """ Returns a string with the current call location. """
    f = inspect.currentframe().f_back.f_back
    return f"{f.f_code.co_filename}:{f.f_lineno}"


def _get_qualified_name(item: Any) -> str:
    """ Returns the name of a decorated item. """
    if inspect.ismethod(item):
        return f"{item.__class__.__name__}"

    name_split = item.__qualname__.split(".")
    if len(name_split) > 1 and name_split[1] in ["__init__", "__new__"]:
        return name_split[0]

    return item.__qualname__


def deprecated(decorated_item: Optional[Any] = None,
               *,
               version: Optional[str] = None,
               instructions: Optional[str] = None,
               warn_once: bool = True) -> Any:
    """ Decorator for marking functions, methods or classes deprecated.

    This decorator logs a deprecation warning whenever the decorated item is
    called. It has the following format:

        From {call info}: {function/method/class} (from {module}) is deprecated
        and will be removed in the future.
        Instructions for updating: {instructions}

    The field {function/method/class} will contain:

        * the function's name, if the decorated item is a function;
        * the method's name and the method's class name, if the decorated item
          is a method;
        * the class' name, if the decorated item is a class.

    This decorator also edits the docstring of the decorated item. A deprecation
    notice is added to the start of the docstring.

    Args:
        decorated_item (Optional[Any]): The item being decorated. Having this
            parameter allows the decorator to be used without arguments.
        instructions (Optional[str]): Instructions on how to update the code
            using the deprecated item.
        version (Optional[str]): Version in which the item was deprecated.
        warn_once (bool): If `True`, a warning will be printed only the first
            time the decorated item is called. Otherwise, every call will log a
            warning.

    Returns:
        The decorated function, method or class.
    """
    version = "" if version is None else f" since version {version}"
    instructions = ("" if instructions is None
                    else f"Instructions for updating: {instructions}")

    def deprecated_wrapper(func_or_class):
        """ Wraps the `deprecated` function, allowing it to accept more
        arguments.
        """
        if inspect.isclass(func_or_class):
            # If a class is deprecated, we actually want to wrap its constructor
            cls = func_or_class
            func = cls.__init__
            constructor_name = "__init__"
        else:
            cls = None
            constructor_name = None
            func = func_or_class

        @functools.wraps(func)
        def func_wrapper(*args, **kwargs):
            """ Wraps the function being decorated. """
            if _PRINT_DEPRECATION_WARNINGS:
                if func not in _PRINTED_WARNINGS_COUNT:
                    _PRINTED_WARNINGS_COUNT[func] = 0

                if _PRINTED_WARNINGS_COUNT[func] == 0 or not warn_once:
                    _PRINTED_WARNINGS_COUNT[func] += 1
                    logging.warning(
                        f"From {_get_call_location()}: "
                        f"`{_get_qualified_name(func)}` (from "
                        f"{func.__module__}) is deprecated{version} and will "
                        f"be removed in the future.\n{instructions}"
                    )
            return func(*args, **kwargs)

        deprecated_msg = f"  ``Deprecated{version}.``"
        if instructions != "":
            deprecated_msg += f"\n\n    {instructions}"
        deprecated_msg += "\n\n   "

        if cls is None:
            # Updating the docstring of the function/method
            func_wrapper.__doc__ = deprecated_msg + func_wrapper.__doc__

            return func_wrapper
        else:
            # Making the function wrapper the new constructor:
            setattr(cls, constructor_name, func_wrapper)

            # Updating the docstring of the class:
            cls.__doc__ = deprecated_msg + cls.__doc__

            return cls

    if decorated_item is None:
        return deprecated_wrapper

    return deprecated_wrapper(decorated_item)

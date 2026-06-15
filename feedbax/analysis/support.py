"""Support helpers for analysis specifications and setup."""

from collections.abc import Callable
from dataclasses import dataclass, fields
import functools
import inspect
import logging
from pathlib import Path
import re
import types
from typing import Any, Optional, get_origin

import jax.tree as jt
import jax_cookbook.tree as jtree
from jax_cookbook import is_type
from jax_cookbook._func import wrap_to_accept_var_kwargs

logger = logging.getLogger(__name__)


def camel_to_snake(s: str):
    """Convert camel case to snake case."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", s).lower()


def get_dataclass_fields(
    obj: Any,
    exclude: tuple[str, ...] = (),
    include_internal: bool = False,
) -> dict[str, Any]:
    """Get the fields of a dataclass object as a dictionary."""
    return {
        field.name: getattr(obj, field.name)
        for field in fields(obj)
        if field.name not in exclude
        and (include_internal or not field.metadata.get("internal", False))
    }


def is_json_serializable(value):
    """Recursively check whether a value can be represented as JSON."""
    json_types = (str, int, float, bool, type(None))

    if isinstance(value, json_types):
        return True
    if isinstance(value, dict):
        return all(isinstance(k, str) and is_json_serializable(v) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return all(is_json_serializable(item) for item in value)
    return False


def field_names(datacls) -> tuple[str, ...]:
    return tuple(datacls.__dataclass_fields__)


def get_origin_type(type_):
    """Get the origin type of a generic type, or the type itself if not generic."""
    origin = get_origin(type_)
    return origin if origin is not None else type_


PATH_DELIM = "`"


def path_delim(p: Path | str) -> str:
    return f"{PATH_DELIM}{str(p)}{PATH_DELIM}"


def location_inspect(fn) -> tuple[str, str, int]:
    """Return a best-effort source location tuple for a function."""
    mod = fn.__module__
    try:
        srcfile = inspect.getsourcefile(fn) or inspect.getfile(fn)
        _, start = inspect.getsourcelines(fn)
        return mod, srcfile, start
    except (OSError, TypeError):
        return mod, fn.__code__.co_filename, fn.__code__.co_firstlineno


def location_for_log(fn) -> str:
    """Return a compact function source location for logs."""
    mod, srcfile, start = location_inspect(fn)
    return f"{mod} ({path_delim(srcfile)}:{start})"


def get_name_of_callable(
    func: Callable,
    *,
    logger: logging.Logger = logger,
) -> str:
    """Return a stable-ish display name for a callable."""
    func_name = getattr(func, "__name__", None)

    if func_name == "<lambda>":
        lambda_loc_str = location_for_log(func)
        logger.debug(f"Assigned name 'lambda' to lambda function at {lambda_loc_str}")
        return "lambda"
    if isinstance(func, functools.partial):
        return get_name_of_callable(func.func, logger=logger)
    if inspect.ismethod(func):
        if hasattr(func, "__self__"):
            return f"{func.__self__.__class__.__name__}.{func.__name__}"
        return func.__name__
    if callable(func) and not isinstance(
        func,
        (types.FunctionType, types.BuiltinFunctionType, type),
    ):
        class_name = func.__class__.__name__
        logger.warning(
            f"Generating name for instance of callable class '{class_name}'. "
            f"Note that instance attributes/state are not captured by this name."
        )
        return class_name
    if func_name is not None:
        return func.__name__
    return repr(func)


def take_model(*args, **kwargs):
    """Apply `jtree.take` to a Feedbax model while treating intervenors as leaves."""
    from feedbax.intervene import AbstractIntervenor

    return jtree.filter_wrap(
        lambda x: not is_type(AbstractIntervenor)(x),
        is_leaf=is_type(AbstractIntervenor),
    )(jtree.take)(*args, **kwargs)


@dataclass
class _OptionalCallableFieldConverter[T]:
    """Converter for dataclass fields that may be callables, constants, or `None`."""

    name: str
    kw_only: bool = True
    wrap_for_var_kwargs: bool = True

    def __call__(self, x: T | Callable[..., T] | None) -> Optional[Callable[..., T]]:
        if x is None:
            return None
        if isinstance(x, Callable):
            if self.kw_only:
                sig = inspect.signature(x)
                if any(
                    p.kind is not inspect.Parameter.KEYWORD_ONLY for p in sig.parameters.values()
                ):
                    raise ValueError(
                        f"Callables assigned to {self.name} cannot have positional parameters."
                    )
            if self.wrap_for_var_kwargs:
                return wrap_to_accept_var_kwargs(x)
            return x
        return lambda **kwargs: x

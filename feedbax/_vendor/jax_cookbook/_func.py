# Modified by Feedbax: internal imports use the bundled package namespace.
import functools
import hashlib
import inspect
import logging
import pickle
import re
import types
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from functools import reduce, wraps
from operator import not_
from typing import Any, Generic, ParamSpec, Set, Tuple, TypeVar

import jax.tree as jt

from feedbax._vendor.jax_cookbook.misc import _fname

logger = logging.getLogger(__name__)


P = ParamSpec("P")
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")

Pred = Callable[P, bool]


def falsef(x: Any) -> bool:
    return False


def truef(x: Any) -> bool:
    return True


def anyf(func: Pred, *funcs: Pred) -> Pred:
    """Logical OR of boolean predicates with the same signature.

    This is useful when we want to satisfy any of a number of `is_leaf`-like conditions
    without writing another ugly lambda. For example:

        `is_leaf=lambda x: is_module(x) or eqx.is_array(x)`

    becomes `is_leaf=anyf(is_module, eqx.is_array)`.
    """
    preds = (func,) + funcs

    def inner(*args: P.args, **kwargs: P.kwargs) -> bool:
        return any(f(*args, **kwargs) for f in preds)

    # Metadata that makes sense for a composite
    inner.__name__ = "anyf"
    inner.__doc__ = "Returns True if any of: " + ", ".join(_fname(f) for f in preds)
    try:
        # Helpful for IDE call tips
        inner.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    except Exception:
        pass
    functools.update_wrapper(inner, func, assigned=(), updated=())
    return inner


def allf(func: Pred, *funcs: Pred) -> Pred:
    """Logical AND of boolean predicates with the same signature.

    This is useful when we want to satisfy any of a number of `is_leaf`-like conditions
    without writing another ugly lambda. For example:

        `is_leaf=lambda x: is_module(x) and eqx.is_array(x)`

    becomes `is_leaf=allf(is_module, eqx.is_array)`.
    """
    preds = (func,) + funcs

    def inner(*args: P.args, **kwargs: P.kwargs) -> bool:
        return all(f(*args, **kwargs) for f in preds)

    # Metadata that makes sense for a composite
    inner.__name__ = "allf"
    inner.__doc__ = "Returns True if all of: " + ", ".join(_fname(f) for f in preds)
    try:
        # Helpful for IDE call tips
        inner.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    except Exception:
        pass
    functools.update_wrapper(inner, func, assigned=(), updated=())
    return inner


def notf(func: Pred) -> Pred:
    """Returns a function that returns the negation of the input function."""

    def inner(*args: P.args, **kwargs: P.kwargs) -> bool:
        return not func(*args, **kwargs)

    # Metadata that makes sense for a composite
    inner.__name__ = f"not_{_fname(func)}"
    inner.__doc__ = f"Returns False if {_fname(func)} returns True."
    try:
        # Helpful for IDE call tips
        inner.__signature__ = inspect.signature(func)  # type: ignore[attr-defined]
    except Exception:
        pass
    functools.update_wrapper(inner, func, assigned=(), updated=())
    return inner


def compose_(*funcs):
    """Compose functions from left to right.

    Args:
        *funcs: Functions to compose, applied left to right (first to last)

    Returns:
        Composite function, whose arguments are those of the first function in `funcs`,
        and whose returns are those of the last.
    """
    if len(funcs) == 1:
        return funcs[0]

    def composite(f, g):
        return lambda *args, **kwargs: g(f(*args, **kwargs))

    return reduce(composite, funcs)


class ComposedFn(Generic[P, T_co]):
    """Builds left-to-right function compositions via `.then()`."""

    def __init__(self, f: Callable[P, T_co]):
        self._fs: list[Callable[..., object]] = [f]  # runtime only

    def then(self, g: Callable[[T_co], U]) -> "ComposedFn[P, U]":
        self._fs.append(g)  # type: ignore[arg-type]
        return self  # type: ignore[return-value]

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T_co:  # type: ignore[misc]
        val = self._fs[0](*args, **kwargs)
        for f in self._fs[1:]:
            val = f(val)
        return val  # type: ignore[return-value]


def compose(f: Callable[P, T_co]) -> ComposedFn[P, T_co]:
    """Alternative to `compose` that maintains the correct parameter and return types for the composite.

    For example, `fn = compose_(f).then(g).then(h)` will type check as a `ComposedFn` whose generic
    parameters are the parameter spec of `f` and the return type of `h`; in an IDE, if we attempt
    to call `fn(...)`, we will see the signature of `f` and the return type of `h`.
    """
    return ComposedFn(f)


def bundle(*funcs):
    """Bundle functions into a single function that returns a tuple of their results."""

    def _funcs(*args, **kwargs):
        return tuple(f(*args, **kwargs) for f in funcs)

    return _funcs


def is_type(*types_) -> Callable[[Any], bool]:
    """Returns a function that returns `True` if the input is an instance of any of the given types."""
    return lambda x: any(isinstance(x, t) for t in types_)


def is_not_type(*types) -> Callable[[Any], bool]:
    """Returns a function that returns `True` if the input is not an instance of any of the given types."""
    return lambda x: not is_type(*types)(x)


def identity(x):
    """The identity function."""
    return x


_VAR_KW_PARAM = inspect.Parameter(
    "__kwargs",
    kind=inspect.Parameter.VAR_KEYWORD,
    annotation=Mapping[str, Any],  # optional; omit if you don't care
)


def wrap_to_accept_var_kwargs(func: Callable, *, strict=False, allowed_extra: Sequence[str] = ()):
    """
    Wrap `func` so you can pass arbitrary **kwargs.
    Unknown kwargs are dropped by default; set strict=True to raise instead.

    TODO: I think this fails in the particular case that `func` has a positional arg with the
    TODO: same name as one of the `kwargs` that the wrapper ends up taking. Since we'll
    TODO: infer "ah, we should include `some_var`, it is one of the arg names" but actually this
    TODO: only applies if it's a kwarg.
    """
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        # Builtins or weird callables we can't introspect.
        if strict:
            raise ValueError(f"Cannot introspect {func} to determine keyword arguments.") from e
        logger.warning(
            f"Cannot introspect {func} to determine keyword arguments. "
            "Returning unwrapped function."
        )
        return func

    params = sig.parameters.values()
    has_varkw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params)
    if has_varkw:
        return func

    allowed = {
        name
        for name, p in sig.parameters.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    } | set(allowed_extra)

    @wraps(func)
    def wrapped(*args, **kwargs):
        if not kwargs:
            return func(*args)
        if strict:
            unknown = set(kwargs) - allowed
            if unknown:
                unknown_list = ", ".join(sorted(unknown))
                raise TypeError(f"{func.__name__}() got unexpected keyword(s): {unknown_list}")
        filtered = {k: v for k, v in kwargs.items() if k in allowed}
        return func(*args, **filtered)

    wrapped.__signature__ = sig.replace(parameters=[*sig.parameters.values(), _VAR_KW_PARAM])
    return wrapped


def _walk(v: Any, _seen: Set[int] | None = None) -> list[Any]:
    """
    Yield *v* itself **plus** all its PyTree leaves (if any),
    with duplicates removed (by object identity).
    """
    if _seen is None:
        _seen = set()
    if id(v) in _seen:
        return []
    _seen.add(id(v))

    try:
        items = jt.leaves(v)
    except ValueError as exc:
        if not isinstance(v, Mapping) or "sorting pytree dictionary keys" not in str(exc):
            raise
        ordered_items = sorted(v.items(), key=lambda item: _bytes_for_constant(item[0]))
        items = [_bytes_for_constant(key) for key, _ in ordered_items]
        for _, value in ordered_items:
            items.extend(_walk(value, _seen))
    else:
        if isinstance(v, Mapping):
            try:
                ordered_keys = sorted(v)
            except TypeError:
                ordered_keys = sorted(v, key=_bytes_for_constant)
            items = [_bytes_for_constant(key) for key in ordered_keys] + items

    if callable(v):
        items.append(v)

    uniq, seen = [], set()
    for obj in items:
        oid = id(obj)
        if oid not in seen:
            seen.add(oid)
            uniq.append(obj)
    return uniq


# --------------------------------------------------------------------------- #
# -- utility helpers                                                          #
# --------------------------------------------------------------------------- #
def _norm_code(co: types.CodeType) -> types.CodeType:
    """Strip line-number metadata so identical logic hashes the same."""
    try:
        kwargs = {"co_firstlineno": 0}
        if hasattr(co, "co_linetable"):
            kwargs["co_linetable"] = b""
        elif hasattr(co, "co_lnotab"):
            kwargs["co_lnotab"] = b""
        return co.replace(**kwargs)
    except Exception:  # pragma: no cover
        return co


def _bytes_for_constant(obj: Any) -> bytes:
    """Deterministic bytes for a non-callable constant."""
    try:
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception:  # pragma: no cover
        return repr(obj).encode()


# --------------------------------------------------------------------------- #
# -- private collector ------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def _collect_parts(
    obj: Any,
    parts: OrderedDict[str, bytes] | None = None,
    *,
    ignore: Tuple[Callable, ...] = (),
    _memo_obj: Set[int] | None = None,
    _memo_code: Set[int] | None = None,
) -> OrderedDict[str, bytes]:
    """
    Populate *parts* with section-key → raw-bytes for everything that
    influences the behaviour of *obj* (callable or code object).
    """
    if parts is None:
        parts = OrderedDict()
    if _memo_obj is None:
        _memo_obj = set()
    if _memo_code is None:
        _memo_code = set()
    assert parts is not None

    ignore_types = tuple(x for x in ignore if isinstance(x, type))

    if obj in ignore or isinstance(obj, ignore_types):
        # Treat as opaque constant: just pickle / repr its identity once
        if id(obj) not in _memo_obj:
            parts[f"ignored:{id(obj)}"] = _bytes_for_constant(obj)
            _memo_obj.add(id(obj))
        return parts

    # -- code objects ---------------------------------------------------- #
    if isinstance(obj, types.CodeType):
        if id(obj) in _memo_code:
            return parts
        _memo_code.add(id(obj))

        co = _norm_code(obj)
        parts[f"code:{id(co)}"] = co.co_code

        for idx, const in enumerate(co.co_consts):
            if isinstance(const, types.CodeType) or callable(const):
                for leaf in _walk(const):
                    _collect_parts(
                        leaf,
                        parts,
                        _memo_obj=_memo_obj,
                        _memo_code=_memo_code,
                        ignore=ignore,
                    )
            else:
                parts[f"const:{id(co)}:{idx}"] = _bytes_for_constant(const)
        return parts

    # non-callables do nothing
    if not callable(obj):
        return parts

    # -- callables (functions, methods, callable instances) -------------- #
    if id(obj) in _memo_obj:
        return parts
    _memo_obj.add(id(obj))

    if isinstance(obj, types.FunctionType):
        func, inst_state = obj, None
    elif inspect.ismethod(obj):
        func, inst_state = obj.__func__, None
    elif hasattr(obj, "__call__"):
        bound = obj.__call__
        func = getattr(bound, "__func__", bound)
        inst_state = getattr(obj, "__dict__", None)

        if not hasattr(func, "__code__"):
            parts[f"opaque-callable:{id(obj)}"] = (
                f"{type(obj).__module__}.{type(obj).__qualname__}".encode()
            )
            return parts
    else:  # pragma: no cover
        raise TypeError(f"{obj!r} is not a pure-Python callable")

    # 1) byte-code + nested constants ------------------------------------ #
    _collect_parts(func.__code__, parts, _memo_obj=_memo_obj, _memo_code=_memo_code, ignore=ignore)

    # 2) decorators ------------------------------------------------------- #
    deco_bytes = b""
    try:
        src, _ = inspect.getsourcelines(func)
        pre_def = []
        for line in src:
            s = line.lstrip()
            if s.startswith("@"):
                pre_def.append(line)
            elif s.startswith(("def ", "async def ")):
                break
            else:
                break
        if pre_def:
            deco_bytes = re.sub(r"\s+", "", "".join(pre_def)).encode()
    except (OSError, IOError):
        pass
    parts[f"decorators:{id(func)}"] = deco_bytes

    # 3) defaults --------------------------------------------------------- #
    defaults_seq = (func.__defaults__ or ()) + tuple(
        (getattr(func, "__kwdefaults__", {}) or {}).values()
    )
    if defaults_seq:
        blob = bytearray()
        for v in defaults_seq:
            for leaf in _walk(v):
                _collect_parts(
                    leaf,
                    parts,
                    _memo_obj=_memo_obj,
                    _memo_code=_memo_code,
                    ignore=ignore,
                )
                if not callable(leaf):
                    blob.extend(_bytes_for_constant(leaf))
        parts[f"defaults:{id(func)}"] = bytes(blob)

    # 4) closure cells (include names only when >1 cell) ------------------ #
    if func.__closure__:
        blob = bytearray()
        include_names = len(func.__closure__) > 1
        for name, cell in zip(func.__code__.co_freevars, func.__closure__):
            if include_names:
                blob.extend(name.encode())
            val = cell.cell_contents
            for leaf in _walk(val):
                _collect_parts(
                    leaf,
                    parts,
                    _memo_obj=_memo_obj,
                    _memo_code=_memo_code,
                    ignore=ignore,
                )
                if not callable(leaf):
                    blob.extend(_bytes_for_constant(leaf))
        parts[f"closure:{id(func)}"] = bytes(blob)

    # 5) names referenced in byte-code (attributes, globals, etc.) --------
    if func.__code__.co_names:
        blob = bytearray()
        g = func.__globals__
        for name in func.__code__.co_names:  # keep order as-compiled
            blob.extend(name.encode())  # always hash the identifier
            if name in g:  # plus its global value if it exists
                val = g[name]
                for leaf in _walk(val):
                    _collect_parts(
                        leaf,
                        parts,
                        _memo_obj=_memo_obj,
                        _memo_code=_memo_code,
                        ignore=ignore,
                    )
                    if not callable(leaf) and not isinstance(leaf, types.CodeType):
                        blob.extend(_bytes_for_constant(leaf))
        parts[f"names:{id(func)}"] = bytes(blob)

    # 6) instance state (callable-class) ---------------------------------- #
    if inst_state:
        blob = bytearray()
        for k in sorted(inst_state):
            blob.extend(k.encode())
            v = inst_state[k]
            for leaf in _walk(v):
                _collect_parts(
                    leaf,
                    parts,
                    _memo_obj=_memo_obj,
                    _memo_code=_memo_code,
                    ignore=ignore,
                )
                if not callable(leaf):
                    blob.extend(_bytes_for_constant(leaf))
        parts[f"state:{id(obj)}"] = bytes(blob)

    return parts


# --------------------------------------------------------------------------- #
# -- public helpers ---------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def hash_callable(obj: Any, ignore: tuple[Callable, ...] = ()) -> str:
    """Return a single SHA-256 hex digest for *obj*’s behaviour."""
    parts = _collect_parts(obj, ignore=ignore)
    return hashlib.sha256(b"".join(parts.values())).hexdigest()


def fingerprint_details(obj: Any) -> Tuple[str, OrderedDict[str, str]]:
    """
    Return (**digest**, **details**).  *details* maps section-keys to their
    individual SHA-256 digests, useful for diffing.
    """
    parts = _collect_parts(obj)
    digest = hashlib.sha256(b"".join(parts.values())).hexdigest()
    details = OrderedDict((k, hashlib.sha256(v).hexdigest()) for k, v in parts.items())
    return digest, details

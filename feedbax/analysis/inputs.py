"""Input and dependency-specification DSL for analysis nodes."""

import dataclasses
import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Generic, Optional, Self, TypeVar, Union

import equinox as eqx
import jax.tree as jt
import jax.tree_util as jtu
from equinox import Module
from jax_cookbook import is_module, is_none
from jaxtyping import PyTree

from feedbax.analysis.types import AnalysisInputData

if TYPE_CHECKING:
    from feedbax.analysis.analysis import AbstractAnalysis


@dataclass(frozen=True)
class AnalysisRef[T]:
    """A thin, typed pointer to something that will yield a PyTree[T]."""

    target: Union[str, "AbstractAnalysis"]  # name or instance


@dataclass(frozen=True, slots=True)
class _DataField:
    """Description of what to extract from `AnalysisInputData`."""

    attr: str
    where: Optional[Callable] = None
    is_leaf: Optional[Callable[[Any], bool]] = is_module

    def __call__(
        self,
        *,
        where: Optional[Callable] = None,
        is_leaf: Optional[Callable[[Any], bool]] = None,
    ) -> Self:
        """Return a new `_DataField` overriding *where* and/or *is_leaf*."""
        return type(self)(
            self.attr,
            where if where is not None else self.where,
            is_leaf if is_leaf is not None else self.is_leaf,
        )

    def __repr__(self):  # noqa: D401
        return f"Data.{self.attr}"


T = TypeVar("T")


@jtu.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class ExpandTo:
    """Prefix-expand a PyTree input to match another input's structure."""

    target: Union[str, _DataField]
    source: PyTree
    where: Optional[Callable] = None
    is_leaf: Optional[Callable] = None
    is_leaf_prefix: Optional[PyTree[Callable]] = None

    @classmethod
    def map(
        cls,
        target: Union[str, _DataField],
        source: T,
        is_leaf_prefix: Optional[PyTree[Callable]] = None,
        **kwargs,
    ) -> T:
        """Convenience for mapping over the first level of a tree."""
        source_with_is_leaf = jt.map(
            lambda is_leaf_node_prefix, node: (node, is_leaf_node_prefix),
            is_leaf_prefix,
            source,
            is_leaf=is_none,
        )
        nodes_with_is_leaf, treedef = eqx.tree_flatten_one_level(source_with_is_leaf)
        return jt.unflatten(
            treedef,
            [
                cls(target, node, is_leaf_prefix=is_leaf_node_prefix, **kwargs)
                for node, is_leaf_node_prefix in nodes_with_is_leaf
            ],
        )

    def tree_flatten(self):
        return [self.source], SimpleNamespace(
            target=self.target,
            where=self.where,
            is_leaf=self.is_leaf,
            is_leaf_prefix=self.is_leaf_prefix,
        )

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(
            target=aux_data.target,
            source=children[0],
            where=aux_data.where,
            is_leaf=aux_data.is_leaf,
            is_leaf_prefix=aux_data.is_leaf_prefix,
        )


@jtu.register_pytree_node_class
@dataclass(frozen=True, slots=True)
class Transformed:
    """Apply a transformation to resolved dependencies before passing to compute."""

    source: PyTree
    transform: Callable

    def tree_flatten(self):
        return [self.source], (self.transform,)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (transform,) = aux_data
        return cls(source=children[0], transform=transform)

    @classmethod
    def map(cls, source: PyTree, transform: Callable, is_leaf: Optional[Callable] = None) -> PyTree:
        def _map_transform(tree):
            return jt.map(transform, tree, is_leaf=is_leaf)

        return cls(source=source, transform=_map_transform)


@dataclass(frozen=True, slots=True)
class LiteralInput:
    """Wrap a constant value to pass directly as an analysis input."""

    value: Any


class _DataProxy:
    """Expose only valid `AnalysisInputData` attributes."""

    _allowed = tuple(AnalysisInputData.__annotations__.keys())

    def __getattr__(self, item: str) -> _DataField:  # noqa: D401
        if item not in self._allowed:
            raise AttributeError(
                f"'Data' has no attribute '{item}'. "
                f"Valid attributes are: {', '.join(self._allowed)}"
            )
        return _DataField(item)

    def __repr__(self):  # noqa: D401
        return "Data"


Data = _DataProxy()


type InputOf[T] = Union[
    AnalysisRef[PyTree[T]],
    str,
    "AbstractAnalysis",
    _DataField,
    Transformed,
    ExpandTo,
    LiteralInput,
]


class AbstractAnalysisPorts(Module, Mapping):
    """Base class for typed analysis input ports."""

    @classmethod
    def converter(cls, inputs: Self | Mapping[str, Any]) -> Self:
        """Convert inputs to the appropriate format."""
        if isinstance(inputs, Mapping):
            return cls(**inputs)
        else:
            return inputs

    def __getitem__(self, key: str):
        if key in self.__dataclass_fields__:
            return getattr(self, key)
        raise KeyError(key)

    def __iter__(self):
        return (f.name for f in dataclasses.fields(self))

    def __len__(self):
        return len(dataclasses.fields(self))


class NoPorts(AbstractAnalysisPorts):
    """An empty Ports dataclass for analyses with no additional inputs."""

    pass


class SinglePort(AbstractAnalysisPorts, Generic[T]):
    """A Ports dataclass with a single port named 'input'."""

    input: InputOf[T] = eqx.field(kw_only=True)


PortsType = TypeVar("PortsType", bound=AbstractAnalysisPorts)


class CallWithDeps:
    """Plug dependency specs into arbitrary argument positions."""

    _counter = 0

    def __init__(self, *pos_specs: Any, **kw_specs: Any):
        self.pos_specs = pos_specs
        self.kw_specs = kw_specs

    @staticmethod
    def _alloc_port() -> str:
        CallWithDeps._counter += 1
        return f"__cwd_{CallWithDeps._counter:x}"

    def __call__(self, fn: Callable):
        spec_map: dict[str, Any] = {}
        pos_tokens: list[Optional[str]] = []
        kw_ports: dict[str, str] = {}

        def port_for(spec: Any) -> str:
            p = self._alloc_port()
            spec_map[p] = spec
            return p

        for item in self.pos_specs:
            pos_tokens.append(None if item is None else port_for(item))
        for name, item in self.kw_specs.items():
            kw_ports[name] = port_for(item)

        sig = inspect.signature(fn)
        param_names = set(sig.parameters.keys())
        has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        @functools.wraps(fn)
        def wrapper(
            *caller_args: Any,
            _spec_map=spec_map,
            _pos_tokens=tuple(pos_tokens),
            _kw_ports=kw_ports,
            _param_names=param_names,
            _has_varkw=has_varkw,
            **caller_kwargs: Any,
        ) -> Any:
            deps: dict[str, Any] = {}
            for p in _spec_map.keys():
                if p in caller_kwargs:
                    deps[p] = caller_kwargs.pop(p)

            missing = [p for p in _spec_map.keys() if p not in deps]
            if missing:
                raise KeyError(
                    f"Missing dependency ports {missing}; have keys={sorted(deps.keys())}"
                )

            args: list[Any] = []
            it = iter(caller_args)
            for tok in _pos_tokens:
                if tok is None:
                    try:
                        args.append(next(it))
                    except StopIteration:
                        raise TypeError(
                            "Not enough positional arguments from caller "
                            "to satisfy `None` placeholders."
                        ) from None
                else:
                    args.append(deps[tok])

            args.extend(list(it))

            mapped_kwargs: dict[str, Any] = dict(caller_kwargs)
            for name, port in _kw_ports.items():
                if name not in mapped_kwargs:
                    mapped_kwargs[name] = deps[port]

            if not _has_varkw:
                mapped_kwargs = {k: v for k, v in mapped_kwargs.items() if k in _param_names}

            return fn(*args, **mapped_kwargs)

        wrapper._extra_inputs = spec_map  # type: ignore[attr-defined]
        wrapper._ports = tuple(spec_map.keys())  # type: ignore[attr-defined]
        return wrapper

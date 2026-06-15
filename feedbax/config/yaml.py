from collections.abc import Callable
from enum import Enum
from pathlib import Path

from ruamel.yaml import YAML, nodes
from jax_cookbook import LDict


class _YamlLiteral:
    def __init__(self, value):
        self.value = value


def _construct_literal(loader, node):
    return _YamlLiteral(loader.construct_sequence(node))


def _represent_enum(representer, data: Enum):
    return representer.represent_scalar(f"!{data.__class__.__name__}", str(data._name_).lower())


def _ldict_multi_constructor(constructor, tag_suffix, node):
    # e.g. !LDict:mylabel {k: v}
    if not isinstance(node, nodes.MappingNode):
        raise TypeError("!LDict:* must tag a mapping")
    mapping = constructor.construct_mapping(node, deep=True)
    return LDict(tag_suffix, mapping)


def _ldict_representer(representer, data: LDict):
    return representer.represent_mapping(f"!LDict:{data.label}", data._data)


def _represent_undefined(representer, data):
    return representer.represent_scalar("tag:yaml.org,2002:str", str(data))


def _yaml_include_constructor(loader, node):
    """YAML constructor to include contents of other YAML files.

    Include paths are resolved relative to the including file, using
    `node.start_mark.name` to identify the including file's path.
    """
    include_path = Path(loader.construct_scalar(node))

    including_file = getattr(node.start_mark, "name", "<unknown>")

    if not include_path.is_absolute():
        try:
            base = Path(node.start_mark.name)
            include_dir = Path(base).resolve().parent
        except AttributeError:
            include_dir = Path(".").resolve()

        include_path = (include_dir / include_path).resolve()

    try:
        yaml = get_yaml_loader(typ="safe")
        with include_path.open("r", encoding="utf-8") as f:
            return yaml.load(f) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Included file '{include_path}' not found (from {including_file})."
        ) from e


REPRESENTERS: dict[type, Callable] = {
    LDict: _ldict_representer,
    object: _represent_undefined,
}

CONSTRUCTORS: dict[str, Callable] = {
    "!Literal": _construct_literal,
    "!include": _yaml_include_constructor,
}

MULTI_CONSTRUCTORS: dict[str, Callable] = {
    "!LDict:": _ldict_multi_constructor,
}


def _iter_representers():
    yield from REPRESENTERS.items()
    try:
        from feedbax.analysis.types import Direction, ResponseVar
    except (ImportError, AttributeError):
        return
    yield Direction, _represent_enum
    yield ResponseVar, _represent_enum


def get_yaml_loader(typ="safe") -> YAML:
    """Returns a ruamel.yaml.YAML instance with representers and constructors for custom types."""
    yaml = YAML(typ=typ)
    yaml.default_flow_style = None
    for type_, representer in _iter_representers():
        yaml.representer.add_representer(type_, representer)
    for tag, constructor in CONSTRUCTORS.items():
        yaml.constructor.add_constructor(tag, constructor)
    for prefix, constructor in MULTI_CONSTRUCTORS.items():
        yaml.constructor.add_multi_constructor(prefix, constructor)
    return yaml

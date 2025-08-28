from importlib import resources
from itertools import product
from typing import Any, Dict, List, Literal, Optional, cast

import jax.tree as jt
import jax.tree_util as jtu
from jax_cookbook import is_type
from jax_cookbook.tree import expand_split_keys
from jaxtyping import PyTree
from ruamel.yaml import YAML

from feedbax_experiments.config.yaml import _YamlLiteral, get_yaml_loader
from feedbax_experiments.misc import deep_merge
from feedbax_experiments.plugins import EXPERIMENT_REGISTRY
from feedbax_experiments.plugins.registry import ExperimentRegistry


def _split_modules(selector: str) -> list[str]:
    return [s.strip() for s in selector.split(",") if s.strip()]


def _node_desc(node: Dict[str, Any]) -> str:
    t = node.get("type", "?")
    name = node.get("name")
    return f"{t} '{name}'" if name else f"{t} (unnamed)"


def _here(node: Dict[str, Any], parent_ctx: Optional[str]) -> str:
    me = _node_desc(node)
    return f"{me} under {parent_ctx}" if parent_ctx else me


def _bad(msg: str, node: Dict[str, Any], parent_ctx: Optional[str]) -> ValueError:
    return ValueError(f"{msg} [at {_here(node, parent_ctx)}]")


def _collect_nonliteral_lengths(m: Dict[str, Any], lengths: set[int]) -> None:
    for v in m.values():
        if isinstance(v, _YamlLiteral):
            continue
        if isinstance(v, list):
            lengths.add(len(v))
        elif isinstance(v, dict):
            _collect_nonliteral_lengths(v, lengths)


def _materialize_index(x: Any, i: int) -> Any:
    if isinstance(x, _YamlLiteral):
        return x.value
    if isinstance(x, list):
        idx = 0 if len(x) == 1 else i
        return _materialize_index(x[idx], i)
    if isinstance(x, dict):
        return {k: _materialize_index(v, i) for k, v in x.items()}
    return x


def _unwrap_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Unwrap literals inside a config mapping, preserving dict[str, Any] type."""

    def _uw(x: Any) -> Any:
        if isinstance(x, dict):
            return {k: _uw(v) for k, v in x.items()}
        if isinstance(x, _YamlLiteral):
            return x.value
        if isinstance(x, list):
            return x
        return x

    # Input is a dict, and we map its values; type-narrow back to Dict[str, Any]
    return cast(Dict[str, Any], _uw(cfg))


def _expand_config_mapping(cfg: Dict[str, Any], ctx: str) -> List[Dict[str, Any]]:
    """
    Expand a config leaf:
      - Dotted keys expanded via `expand_split_keys`
      - Non-literal lists are zipped/broadcast
      - If >1 distinct non-trivial lengths appear, raise (ask for explicit product)
    """
    # 1) Dotted-key expansion returns a generic PyTree; narrow back to Dict[str, Any]
    cfg = cast(Dict[str, Any], expand_split_keys(cfg))

    lengths: set[int] = set()
    _collect_nonliteral_lengths(cfg, lengths)
    lengths = {L for L in lengths if L > 1}

    if not lengths:
        # 2) Return a Dict[str, Any], not Any
        return [_unwrap_config(cfg)]

    if len(lengths) > 1:
        raise ValueError(
            f"Mismatched sweep lengths {sorted(lengths)} inside {ctx}. "
            f"Use an explicit 'product' to combine axes."
        )

    n = next(iter(lengths))
    out: List[Dict[str, Any]] = []
    for i in range(n):
        mat = _materialize_index(cfg, i)
        # Defensive runtime check + narrow for the type checker
        if not isinstance(mat, dict):
            raise TypeError(
                f"Config expansion produced non-mapping at index {i}: {type(mat).__name__}"
            )
        out.append(cast(Dict[str, Any], mat))
    return out


def _eval_node(node: Dict[str, Any], parent_ctx: Optional[str]) -> List[Dict[str, Any]]:
    t = node.get("type")
    if t not in {"config", "product", "cases"}:
        raise _bad(f"Unknown node type: {t!r}", node, parent_ctx)

    if t == "config":
        cfg = node.get("of", {})
        if not isinstance(cfg, dict):
            raise _bad("config node expects a mapping under 'of'", node, parent_ctx)
        if any(k in cfg for k in ("type", "product", "cases")):
            raise _bad(
                (
                    "composition keys ('type', 'product', 'cases') are not allowed "
                    "inside a config leaf"
                ),
                node,
                parent_ctx,
            )
        return _expand_config_mapping(cfg, ctx=_here(node, parent_ctx))

    if t == "cases":
        children = node.get("of", [])
        if not isinstance(children, list):
            raise _bad("cases node expects a list under 'of'", node, parent_ctx)
        out: List[Dict[str, Any]] = []
        for child in children:
            out.extend(_eval_node(child, parent_ctx=_here(node, parent_ctx)))
        return out

    elif t == "product":
        children = node.get("of", [])
        if not isinstance(children, list):
            raise _bad("product node expects a list under 'of'", node, parent_ctx)
        axes = [_eval_node(child, parent_ctx=_here(node, parent_ctx)) for child in children]
        out: List[Dict[str, Any]] = []
        for tpl in product(*axes):
            merged: Dict[str, Any] = {}
            for d in tpl:
                merged = deep_merge(merged, d)  # <- your deep_merge
            out.append(merged)
        return out

    else:
        assert False


def load_batch_config(
    domain: Literal["analysis", "training"],
    config_key: str,
    registry: Optional[ExperimentRegistry] = None,
) -> dict[str, list[dict]]:
    """
    Load a batched config file and return { module_key: [run_params, ...] }.

    Addressing (config_key):
      - "pkg/name" or "name"
        * If unqualified and multiple packages are registered, resolution is ambiguous.
        * If unqualified and exactly one package registered, use it.
        * Else probe packages for a unique match of '{pkg}.config.batched.{domain}/{name}.yml'.

    Node semantics (unchanged):
      - type=config: zipped/broadcast sweeps inside 'of' (no dotted keys in output)
      - type=product: cartesian product over children (deep-merge)
      - type=cases:   union of children
    """
    if registry is None:
        registry = EXPERIMENT_REGISTRY

    yaml = get_yaml_loader(typ="safe")

    # Resolve owning package for this batch key (supports "pkg/name" or "name")
    if hasattr(registry, "resolve_package_for_batch_key"):
        pkg = registry.resolve_package_for_batch_key(config_key, domain=domain)
        name = config_key.split("/", 1)[1] if "/" in config_key else config_key
    else:
        # Fallback resolution (in case helper isn't implemented yet)
        if "/" in config_key:
            pkg, name = config_key.split("/", 1)
            if pkg not in registry._packages:
                raise ValueError(f"Package '{pkg}' not found in registry")
        else:
            single = registry.single_package_name()
            if single:
                pkg, name = single, config_key
            else:
                # Probe for unique match
                matches: list[str] = []
                for pkg_name, md in registry._packages.items():
                    root = (
                        f"{md.package_module.__name__}.{md.config_resource_root}.batched.{domain}"
                    )
                    try:
                        if resources.files(root).joinpath(f"{config_key}.yml").is_file():
                            matches.append(pkg_name)
                    except Exception:
                        pass
                if not matches:
                    raise FileNotFoundError(
                        f"Batch config '{config_key}.yml' not found under any package's config.batched.{domain}"
                    )
                if len(matches) > 1:
                    opts = "', '".join(matches)
                    raise ValueError(
                        f"Batch key '{config_key}' is ambiguous across packages ({opts}). "
                        f"Use '<package>/{config_key}'."
                    )
                pkg, name = matches[0], config_key

    # Open exactly one batch file under that package
    md = registry._packages[pkg]
    resource_root = f"{md.package_module.__name__}.{md.config_resource_root}.batched.{domain}"

    try:
        with resources.open_text(resource_root, f"{name}.yml", encoding="utf-8") as f:
            doc = yaml.load(f) or {}
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise FileNotFoundError(f"Batch config '{name}.yml' not found under {resource_root}") from e

    spec = doc.get("SPEC")
    if spec is None:
        raise ValueError("Batch config file must contain a top-level 'SPEC' key.")
    if not isinstance(spec, list):
        raise ValueError("Top-level 'SPEC' must be a list of module-spec entries.")

    out: Dict[str, List[Dict[str, Any]]] = {}

    for idx, entry in enumerate(spec, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"SPEC item #{idx} must be a mapping of 'modules: node'.")
        if len(entry) == 0:
            continue  # allow empty entries
        # Support multiple keys in a single mapping item if you like
        for selector, node in entry.items():
            if not isinstance(selector, str):
                raise ValueError(
                    f"SPEC item #{idx} key must be a string of one or more comma-separated "
                    "module names."
                )
            modules = _split_modules(selector)
            if not modules:
                raise ValueError(f"SPEC item #{idx} has empty module selector.")
            if not isinstance(node, dict):
                raise ValueError(f"SPEC item #{idx} value must be a typed node mapping.")

            for modkey in modules:
                runs = _eval_node(node, parent_ctx=f"module '{modkey}' via SPEC item #{idx}")
                out.setdefault(modkey, []).extend(runs)

    return out

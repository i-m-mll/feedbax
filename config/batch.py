from importlib import resources
from itertools import product
from typing import Any, Dict, List, Literal, Optional, cast

import jax.tree as jt
import jax.tree_util as jtu
import yaml
from jax_cookbook import is_type
from jaxtyping import PyTree

from rlrmp.misc import deep_merge

# from rlrmp.tree_utils import expand_split_keys


def expand_split_keys(tree: PyTree, key_sep: str = ".") -> PyTree:
    """Expand dotted keys inside mappings into nested dicts, recursively."""
    if isinstance(tree, dict):
        out = {}
        for key, val in tree.items():
            val = expand_split_keys(val, key_sep=key_sep)
            if isinstance(key, str) and key_sep in key:
                print(key)
                parts = key.split(key_sep)
                cur = out
                for p in parts[:-1]:
                    if p not in cur or not isinstance(cur[p], dict):
                        cur[p] = {}
                    cur = cur[p]
                last = parts[-1]
                if isinstance(val, dict) and isinstance(cur.get(last), dict):
                    cur[last] = deep_merge(cur[last], val)
                else:
                    cur[last] = val
            else:
                if isinstance(val, dict) and isinstance(out.get(key), dict):
                    out[key] = deep_merge(out[key], val)
                else:
                    out[key] = val
        return type(tree)(**out)
    else:
        leaves, treedef = jt.flatten(tree, is_leaf=is_type(dict))
        if jtu.treedef_is_leaf(treedef) and not isinstance(leaves, dict):
            return tree
        return jt.unflatten(treedef, [expand_split_keys(leaf, key_sep=key_sep) for leaf in leaves])


class _YamlLiteral(list): ...


def _construct_literal(loader: yaml.SafeLoader, node: yaml.nodes.SequenceNode):
    return _YamlLiteral(loader.construct_sequence(node))


yaml.SafeLoader.add_constructor("!Literal", _construct_literal)


def _node_desc(node: Dict[str, Any]) -> str:
    t = node.get("type", "?")
    name = node.get("name")
    return f"{t} '{name}'" if name else f"{t} (unnamed)"


def _here(node: Dict[str, Any], parent_ctx: Optional[str]) -> str:
    me = _node_desc(node)
    return f"{me} under {parent_ctx}" if parent_ctx else me


def _bad(msg: str, node: Dict[str, Any], parent_ctx: Optional[str]) -> ValueError:
    return ValueError(f"{msg} [at {_here(node, parent_ctx)}]")


def _is_literal_list(x: Any) -> bool:
    return isinstance(x, _YamlLiteral)


def _unwrap_literal(x: Any) -> Any:
    return list(x) if isinstance(x, _YamlLiteral) else x


def _collect_nonliteral_lengths(m: Dict[str, Any], lengths: set[int]) -> None:
    for v in m.values():
        if _is_literal_list(v):
            continue
        if isinstance(v, list):
            lengths.add(len(v))
        elif isinstance(v, dict):
            _collect_nonliteral_lengths(v, lengths)


def _materialize_index(x: Any, i: int) -> Any:
    if _is_literal_list(x):
        return _unwrap_literal(x)
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
        if _is_literal_list(x):
            return _unwrap_literal(x)
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
) -> dict[str, list[dict]]:
    """
    Load src/rlrmp/config/batched/{domain}/{config_key}.yml and return:
        { module_key: [run_params, ...] }

    Node semantics:
      - type=config: zipped/broadcast sweeps inside 'of' (no dotted keys in output)
      - type=product: cartesian product over children (deep-merge)
      - type=cases:   union of children
    """
    try:
        with resources.open_text(f"rlrmp.config.batched.{domain}", f"{config_key}.yml") as f:
            batch = yaml.safe_load(f)
    except (FileNotFoundError, ModuleNotFoundError):
        raise FileNotFoundError(
            f"Batch config file {config_key}.yml not found in rlrmp.config.batched.{domain}"
        )

    if not isinstance(batch, dict):
        raise ValueError("Top-level batch config must be a mapping of module_key -> typed node")

    out: Dict[str, List[Dict[str, Any]]] = {}
    for module_key, module_node in batch.items():
        parent_ctx = f"module '{module_key}'"
        out[module_key] = _eval_node(module_node, parent_ctx=parent_ctx)
    return out

"""Compare an edit's selected runs with the effective reach of its available stores."""

from collections.abc import Collection, Sequence
from dataclasses import dataclass
from hashlib import sha256
from typing import Literal

from pydantic import JsonValue

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.jobs import StudioSavedTrainKey
from feedbax.contracts.project_resolution import RunKey
from feedbax.contracts.studio_scope import StoreKind, StudioStoreChoice


@dataclass(frozen=True)
class EditStore[T]:
    kind: StoreKind
    label: str
    source: str
    runs: frozenset[RunKey]
    value: T


def all_readers_store[T](source: str, runs: frozenset[RunKey], value: T) -> EditStore[T]:
    count = len({run.step for run in runs})
    return EditStore("all_readers", f"All {count} Steps" if count > 1 else "Whole Step", source, runs, value)


def choose_store[T](
    revision: str,
    intent: JsonValue,
    selection: Collection[RunKey],
    preferred: EditStore[T],
    alternatives: Sequence[EditStore[T]] = (),
    *,
    choice: str | None = None,
) -> EditStore[T] | list[StudioStoreChoice] | Literal["unavailable", "stale"]:
    """Resolve a preferred store within the selection, or offer the effective stores at this revision.

    The adapter supplies actual reach from authored provenance. A stable source address
    distinguishes multiple stores of the same kind. It is bound to the replay token
    along with the edit, selection, revision and calculated reach.
    """
    def key(run: RunKey) -> tuple[str, str, str]:
        return run.variant, run.step, run.row

    within_selection = bool(preferred.runs) and preferred.runs <= set(selection)
    if choice is None and within_selection:
        return preferred

    options = {}
    for store in (preferred, *alternatives):
        if not store.runs:
            continue
        runs = sorted(store.runs, key=key)
        store_id = sha256(canonical_json_v2_bytes({
            "revision": revision,
            "intent": intent,
            "selection": [list(key(run)) for run in sorted(set(selection), key=key)],
            "source": store.source,
            "kind": store.kind,
            "runs": [list(key(run)) for run in runs],
        })).hexdigest()
        options[store_id] = (store, StudioStoreChoice(
            id=store_id, kind=store.kind, label=store.label,
            steps=list(dict.fromkeys(run.step for run in runs)),
            runs=[StudioSavedTrainKey(variant=run.variant, step=run.step, row=run.row) for run in runs],
        ))

    if choice is not None:
        if within_selection or choice not in options:
            return "stale"
        return options[choice][0]
    return [option for _, option in options.values()] or "unavailable"

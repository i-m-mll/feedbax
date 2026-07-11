import json
from pathlib import Path

import pytest

from feedbax.contracts.lineage import LineageDag, LineageEvent, LineageParentRef, store_lineage_event


def _hash(char: str) -> str:
    return char * 64


def test_graft_modes_and_superseding_interpretation() -> None:
    original = LineageEvent(
        event_kind="execution",
        execution_hash=_hash("a"),
        parents=[LineageParentRef(execution_hash=_hash("b"))],
    )
    dag = LineageDag([original])
    correction = LineageEvent(
        event_kind="graft_correction",
        execution_hash=_hash("a"),
        parents=[LineageParentRef(execution_hash=_hash("c"))],
        original_event_hash=original.content_hash,
        correction_mode="supersedes_for_interpretation",
    )
    dag.append(correction)
    new_execution = LineageEvent(
        event_kind="graft_correction",
        execution_hash=_hash("d"),
        parents=[LineageParentRef(execution_hash=_hash("c"))],
        original_event_hash=original.content_hash,
        correction_mode="new_execution",
    )
    dag.append(new_execution)
    assert dag.interpreted_parents(_hash("a"))[0].execution_hash == _hash("c")
    assert dag.interpreted_parents(_hash("d"))[0].execution_hash == _hash("c")


def test_append_only_rejects_unknown_original_event_hash() -> None:
    dag = LineageDag()
    event = LineageEvent(
        event_kind="graft_correction",
        execution_hash=_hash("a"),
        original_event_hash=_hash("f"),
        correction_mode="new_execution",
    )
    with pytest.raises(ValueError, match="earlier append-only event"):
        dag.append(event)


def test_store_lineage_event_round_trip(tmp_path: Path) -> None:
    event = LineageEvent(event_kind="execution", execution_hash=_hash("a"))
    artifact = store_lineage_event(event, tmp_path, "event.json")
    stored = json.loads(Path(artifact.uri).read_text(encoding="utf-8"))
    assert stored == event.model_dump(mode="json", exclude_none=True)
    assert artifact.sha256 == event.content_hash


def test_lineage_dag_rejects_cycle() -> None:
    a, b = _hash("a"), _hash("b")
    dag = LineageDag([LineageEvent(event_kind="execution", execution_hash=a)])
    dag.append(LineageEvent(event_kind="execution", execution_hash=b, parents=[LineageParentRef(execution_hash=a)]))
    with pytest.raises(ValueError, match="cycle"):
        dag.append(LineageEvent(event_kind="execution", execution_hash=a, parents=[LineageParentRef(execution_hash=b)]))

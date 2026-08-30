"""Core CLI regressions for explicit bootstrapped registry ownership."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import pytest

import feedbax.__main__ as feedbax_cli
from feedbax.bin import orchestrate
from feedbax.orchestration.collection_recovery import CollectionRecoveryBinding
from feedbax.orchestration.conformance import CheckRegistry
from feedbax.orchestration.input_materialization import InputProviderRootBinding
from feedbax.orchestration.staged_root_custody import StagedRootSnapshotBinding
from feedbax.training.interruption import CancellationDecision


class _UnknownMethodRegistry:
    def resolve_execution(self, *_args: Any, **_kwargs: Any) -> None:
        raise ValueError("/method_ref: unknown method_ref 'tests/unknown/v1'")


class _StructuralTrainingRunSpec:
    method_ref = "tests/unknown/v1"
    method_payload = None
    worker_execution = None

    @staticmethod
    def model_validate(_payload: object) -> _StructuralTrainingRunSpec:
        return _StructuralTrainingRunSpec()


def _bootstrap_state(training_programs: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        bundle=SimpleNamespace(
            conformance_checks=object(),
            training_programs=training_programs or object(),
            drivers=object(),
        ),
        provenance=(),
    )


def _install_unknown_method_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    state = _bootstrap_state(_UnknownMethodRegistry())

    async def compose_application(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
        return state

    monkeypatch.setattr(feedbax_cli, "compose_application", compose_application)
    monkeypatch.setattr(feedbax_cli, "TrainingRunSpec", _StructuralTrainingRunSpec)
    monkeypatch.setattr(feedbax_cli, "_read_json", lambda _path: {})


def test_checkpoint_binding_loader_rejects_unknown_training_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_spec_path = tmp_path / "run-spec.json"
    run_spec_path.write_text("{}", encoding="utf-8")
    bindings_path = tmp_path / "bindings.json"
    bindings_path.write_text(
        json.dumps(
            {
                "schema_id": "feedbax.runtime.checkpoint_fork_plan_bindings",
                "schema_version": "feedbax.runtime.checkpoint_fork_plan_bindings.v1",
                "checkpoint_roots": {},
                "run_specs": {"target": run_spec_path.name},
                "slot_templates": {},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(feedbax_cli, "TrainingRunSpec", _StructuralTrainingRunSpec)

    with pytest.raises(ValueError, match="unknown method_ref 'tests/unknown/v1'"):
        feedbax_cli._load_checkpoint_fork_plan_bindings(
            str(bindings_path),
            _UnknownMethodRegistry(),  # type: ignore[arg-type]
        )


def test_checkpoint_fork_reports_unknown_method_before_checkpoint_io(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _install_unknown_method_boundary(monkeypatch)

    result = feedbax_cli.main(
        [
            "checkpoint",
            "fork",
            "--source",
            "missing-source",
            "--target",
            "run-spec.json:missing-target",
        ]
    )

    assert result == 1
    payload = json.loads(capsys.readouterr().out)
    assert "unknown method_ref 'tests/unknown/v1'" in payload["targets"][0]["error"]


class _ReachedCommandBoundary(RuntimeError):
    pass


def _request_engine_probe(
    request: object,
    *,
    request_path: Path,
    run_set_id: str | None = None,
    interruption_probe: Callable[[], CancellationDecision | None] | None = None,
    input_provider_bindings: tuple[InputProviderRootBinding, ...] = (),
    native_update_budget: int | None = None,
    staged_root_bindings: tuple[StagedRootSnapshotBinding, ...] = (),
    conformance_registry: CheckRegistry,
    plugin_provenance: Sequence[Any],
    registry_bundle: Any,
) -> None:
    del (
        request,
        request_path,
        run_set_id,
        interruption_probe,
        input_provider_bindings,
        native_update_budget,
        staged_root_bindings,
        conformance_registry,
        plugin_provenance,
        registry_bundle,
    )
    raise _ReachedCommandBoundary


@pytest.mark.parametrize("command", ("preflight", "shadow-launch"))
def test_orchestration_request_commands_match_engine_registry_boundary(
    command: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(orchestrate, "_load_assembly_request", lambda _path: object())
    monkeypatch.setattr(orchestrate, "_request_engine", _request_engine_probe)
    monkeypatch.setattr(orchestrate, "_require_provider_free_shadow_request", lambda _request: None)
    args = SimpleNamespace(
        assembly_request="request.json",
        authority_only=False,
        bundle=None,
        bundle_sha256=None,
        run_set_id=None,
        input_provider=[],
        staged_root=[],
        bootstrap_state=_bootstrap_state(),
    )

    with pytest.raises(_ReachedCommandBoundary):
        getattr(orchestrate, f"cmd_{command.replace('-', '_')}")(args)


def test_orchestration_teardown_passes_training_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training_methods = object()

    def run_existing(
        run_set_id: str,
        *,
        stop_after_stage: str | None = None,
        break_stale_lock: bool = False,
        retry_failed_certification: bool = False,
        interruption_probe: Callable[[], CancellationDecision | None] | None = None,
        conformance_registry: CheckRegistry,
        training_method_registry: object,
        driver_registry: object,
        plugin_provenance: Sequence[Any],
        input_provider_bindings: tuple[InputProviderRootBinding, ...] = (),
        collection_recovery_bindings: tuple[CollectionRecoveryBinding, ...] = (),
    ) -> None:
        del (
            run_set_id,
            stop_after_stage,
            break_stale_lock,
            retry_failed_certification,
            interruption_probe,
            conformance_registry,
            driver_registry,
            plugin_provenance,
            input_provider_bindings,
            collection_recovery_bindings,
        )
        assert training_method_registry is training_methods
        raise _ReachedCommandBoundary

    monkeypatch.setattr(orchestrate, "_run_existing", run_existing)
    args = SimpleNamespace(
        run_set="run-set",
        force=False,
        bootstrap_state=_bootstrap_state(training_methods),
    )

    with pytest.raises(_ReachedCommandBoundary):
        orchestrate.cmd_teardown(args)

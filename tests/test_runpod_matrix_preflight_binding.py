from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from feedbax.bin import orchestrate
from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRowLoweringResult,
)
from feedbax.orchestration.assembly import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    DeploymentPolicy,
    DeploymentResourceRequest,
    EnvironmentDeclaration,
    RepoRevision,
    RunBundle,
    SchemaArtifactRef,
    canonical_run_bundle_sha256,
)
from feedbax.orchestration.drivers.runpod import (
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V3,
    RunPodDriverConfig,
    RunPodDriverError,
    RunPodOrchestrationDriver,
)
from feedbax.orchestration import matrix_authority
from feedbax.orchestration.stages import (
    STAGE_PREFLIGHT,
    StageEngine,
    run_authority_preflight_checks,
)
from feedbax.orchestration.state import RunSetStateStore
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
    register_training_run_matrix_compiler,
)


_ISOLATED_GIT_ENV = {
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "LC_ALL": "C",
    "PATH": os.defpath,
}


class RecordingTransport:
    def __init__(self) -> None:
        self.operations: list[str] = []

    def image_exists(self, image: str) -> bool:
        self.operations.append(f"image:{image}")
        return True

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"unexpected provider access: {name}")


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        env=_ISOLATED_GIT_ENV,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _authority_repo(root: Path) -> tuple[Path, str]:
    repo = root / "science"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Feedbax Tests")
    (repo / "authority.txt").write_text("protected\n", encoding="utf-8")
    (repo / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(repo, "add", "authority.txt", "uv.lock")
    _git(repo, "commit", "-m", "protected authority")
    return repo, _git(repo, "rev-parse", "HEAD")


def _matrix_case(
    root: Path,
    *,
    revision: str,
    execution_payload: dict[str, Any] | None = None,
    review_required: bool = False,
) -> tuple[RunAssemblyRequest, AssemblyCompilerRegistry, RunBundle]:
    authored = {
        "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
        "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
        "name": "authenticated matrix",
        "base": {"kind": "inline", "inline": {"gain": 2}},
        "rows": [{"row_id": row, "overrides": []} for row in ("first", "second")],
    }
    authored_bytes = (json.dumps(authored, indent=2) + "\n").encode()
    authored_path = root / "matrix.json"
    authored_path.write_bytes(authored_bytes)
    lockfile = root / "science" / "uv.lock"
    lockfile.write_text("version = 1\n", encoding="utf-8")
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            schema_version=TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            artifact_id="artifact://matrix/authenticated",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=TRAINING_RUN_MATRIX_COMPILER_ID,
            compiler_version=TRAINING_RUN_MATRIX_COMPILER_VERSION,
        ),
        deployment_policy=DeploymentPolicy(
            driver="runpod",
            venue="remote",
            cloud_authorized=True,
            review_required=review_required,
            review_authorized=False,
            resources=DeploymentResourceRequest(gpu_id="RTX", regions=["CA"]),
        ),
        environment=EnvironmentDeclaration(
            python_version="3.12",
            repo_revisions=[RepoRevision(path="science", revision=revision, dirty_allowed=False)],
            lockfile_hashes={"uv.lock": hashlib.sha256(lockfile.read_bytes()).hexdigest()},
            image_id="example.invalid/feedbax@sha256:" + "a" * 64,
            metadata={
                "runpod_local_repos": {"science": str(root / "science")},
                "runpod_protected_refs": {"science": "refs/heads/main"},
            },
        ),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(root / "runs"),
    )
    registry = AssemblyCompilerRegistry()

    def lower(authored_row: Any) -> TrainingRowLoweringResult:
        return TrainingRowLoweringResult(
            execution_payload=execution_payload
            or {
                "schema_id": "example.training_payload",
                "schema_version": "example.training_payload.v1",
                "training_config": {"n_batches": 7},
                "gain": authored_row.payload["gain"],
            },
            lowerer_identities=[{"lowerer_id": "tests.matrix", "lowerer_version": "v1"}],
        )

    register_training_run_matrix_compiler(registry, allow_inline_base=True, row_lowerer=lower)
    context = AssemblyContext(custody_root=root / "runs" / "custody", repo_root=root)
    bundle = assemble_run_bundle(
        request, run_set_id="matrix-run-set", context=context, registry=registry
    )
    return request, registry, bundle


def _scheduled_training_payload(
    *,
    resume_context: dict[str, int] | None,
    optimizer_build_context: dict[str, int] | None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if resume_context is not None:
        metadata["resume_context"] = resume_context
    if optimizer_build_context is not None:
        metadata["optimizer_build_context"] = optimizer_build_context
    return {
        "schema_id": "feedbax.spec.training_run",
        "schema_version": "feedbax.spec.training_run.v2",
        "training_config": {"n_batches": 7},
        "method_payload": {
            "payload": {
                "controller_optimizer": {
                    "type": "adamw",
                    "params": {"weight_decay": 0.0},
                    "lr_schedule": {
                        "kind": "warmup_cosine",
                        "learning_rate_0": 0.1,
                        "total_steps": 3500,
                        "constant_lr_iterations": 500,
                        "warmup_init_fraction": 0.1,
                        "cosine_annealing_alpha": 0.2,
                    },
                }
            }
        },
        "metadata": metadata,
    }


def _matrix_bundle(root: Path, *, revision: str) -> RunBundle:
    return _matrix_case(root, revision=revision)[2]


def _driver(bundle: RunBundle, repo: Path, transport: RecordingTransport):
    return RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="203.0.113.10",
            ssh_port=22,
            image=bundle.environment.image_id or "",
            local_repos={"science": repo},
            remote_repos={"science": "/workspace/science"},
            primary_repo="science",
            protected_refs={"science": "refs/heads/main"},
        ),
        transport=transport,
    )


def _completed_preflight(bundle: RunBundle, repo: Path, root: Path) -> Any:
    store = RunSetStateStore(root / "state.json")
    return StageEngine(bundle=bundle, driver=_driver(bundle, repo, RecordingTransport()), store=store).run(stop_after_stage=STAGE_PREFLIGHT)


def test_matrix_preflight_emits_canonical_v3_without_private_paths(tmp_path: Path) -> None:
    repo, revision = _authority_repo(tmp_path)
    bundle = _matrix_bundle(tmp_path, revision=revision)
    state = _completed_preflight(bundle, repo, tmp_path)
    evidence = state.stage(STAGE_PREFLIGHT).outputs["driver_evidence"]
    assert evidence["schema_version"] == RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V3
    assert evidence["v1"]["schema_version"] == RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION
    binding = evidence["matrix_binding"]
    assert [row["row_id"] for row in binding["rows"]] == ["first", "second"]
    assert [row["locked_training_depth"] for row in binding["rows"]] == [7, 7]
    assert binding["monitor"]["event_paths"] == [f"events/{row}.events.jsonl" for row in ("first", "second")]
    authority = binding["code_authorities"][0]
    assert (authority["repo"], authority["protected_ref"], authority["clean"]) == ("science", "refs/heads/main", True)
    assert authority["declared_revision"] == authority["protected_revision"] == revision
    assert authority["observed_revision"] == revision
    assert str(tmp_path) not in json.dumps(evidence)


def test_top_level_matrix_authority_export_does_not_import_runpod() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from feedbax.orchestration import build_training_run_matrix_authority; "
            "import sys; assert 'feedbax.orchestration.drivers.runpod' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_authority_only_cli_is_isolated_and_matches_real_runpod_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo, revision = _authority_repo(tmp_path)
    request, registry, bundle = _matrix_case(tmp_path, revision=revision)
    request_path = tmp_path / "request.json"
    request_path.write_text(request.model_dump_json(indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(orchestrate, "build_default_assembly_registry", lambda: registry)
    monkeypatch.setattr(
        orchestrate,
        "load_training_method_plugins",
        lambda **_kwargs: pytest.fail("authority-only preflight loaded plugins"),
    )
    monkeypatch.setattr(
        orchestrate,
        "load_runpod_api_key",
        lambda: pytest.fail("authority-only preflight read environment/config/keychain credentials"),
    )
    monkeypatch.setattr(
        orchestrate,
        "_runpod_config_for_bundle",
        lambda _bundle: pytest.fail("authority-only preflight constructed provider config"),
    )
    monkeypatch.setattr(
        orchestrate,
        "RunPodOrchestrationDriver",
        lambda *_args, **_kwargs: pytest.fail(
            "authority-only preflight constructed provider driver"
        ),
    )
    real_run = subprocess.run
    git_calls = 0

    def isolated_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal git_calls
        assert kwargs["env"] == _ISOLATED_GIT_ENV
        git_calls += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(matrix_authority.subprocess, "run", isolated_run)

    assert orchestrate.main(
        [
            "preflight",
            "--authority-only",
            "--run-set-id",
            bundle.run_set_id,
            "--assembly-request",
            str(request_path),
        ]
    ) == 0
    neutral = json.loads(capsys.readouterr().out)
    assert neutral["authority_state"] == "provider_unverified"
    assert "nested_preflight_evidence_sha256" not in neutral
    assert neutral["bundle_sha256"] == canonical_run_bundle_sha256(bundle)
    state = _completed_preflight(bundle, repo, tmp_path)
    provider = state.stage(STAGE_PREFLIGHT).outputs["driver_evidence"]
    for key in ("matrix", "rows", "resolved_inputs", "code_authorities", "monitor", "bundle_sha256"):
        assert provider["matrix_binding"][key] == neutral[key]
    stage = state.stage(STAGE_PREFLIGHT)
    state = state.with_stage(
        STAGE_PREFLIGHT,
        stage.model_copy(update={"outputs": {**stage.outputs, "driver_evidence": neutral}}),
    )
    with pytest.raises(RunPodDriverError, match="matrix PREFLIGHT|invalid shape"):
        _driver(bundle, repo, RecordingTransport()).restore_completed_preflight(bundle, state)
    assert git_calls >= 6


def test_authority_only_cli_authenticates_content_pinned_custom_lowered_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _repo, revision = _authority_repo(tmp_path)
    context = {
        "schedule_origin_step": 12_000,
        "current_step": 12_000,
        "optimizer_count_at_current_step": 12_000,
    }
    _request, _registry, bundle = _matrix_case(
        tmp_path,
        revision=revision,
        execution_payload=_scheduled_training_payload(
            resume_context=context,
            optimizer_build_context=context,
        ),
        review_required=True,
    )
    bundle_path = tmp_path / "assembled-bundle.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    bundle_sha256 = canonical_run_bundle_sha256(bundle)
    monkeypatch.setattr(
        orchestrate,
        "assemble_run_bundle",
        lambda *_args, **_kwargs: pytest.fail("bundle authority re-ran downstream lowering"),
    )
    monkeypatch.setattr(
        orchestrate,
        "load_training_method_plugins",
        lambda **_kwargs: pytest.fail("bundle authority loaded plugins"),
    )
    monkeypatch.setattr(
        orchestrate,
        "load_runpod_api_key",
        lambda: pytest.fail("bundle authority read credentials"),
    )
    monkeypatch.setattr(
        orchestrate,
        "_runpod_config_for_bundle",
        lambda _bundle: pytest.fail("bundle authority constructed provider config"),
    )
    monkeypatch.setattr(
        orchestrate,
        "RunPodOrchestrationDriver",
        lambda *_args, **_kwargs: pytest.fail("bundle authority constructed provider driver"),
    )

    checks = {check.name: check for check in run_authority_preflight_checks(bundle)}
    assert set(checks) == {
        "schema-current",
        "feedbax-revision-pin",
        "row-identity",
        "budget-presence",
        "driver-preconditions",
        "environment-declaration",
        "input-custody-authority",
        "native-output-custody",
        "schedule-realization",
    }
    assert all(check.status == "pass" for check in checks.values())

    assert orchestrate.main(
        [
            "preflight",
            "--authority-only",
            "--run-set-id",
            bundle.run_set_id,
            "--bundle",
            str(bundle_path),
            "--bundle-sha256",
            bundle_sha256,
        ]
    ) == 0
    authority = json.loads(capsys.readouterr().out)
    assert authority["authority_state"] == "provider_unverified"
    assert authority["bundle_sha256"] == bundle_sha256
    assert [row["row_id"] for row in authority["rows"]] == ["first", "second"]
    assert [row["locked_training_depth"] for row in authority["rows"]] == [7, 7]
    assert "nested_preflight_evidence_sha256" not in authority


@pytest.mark.parametrize("invalid_context", ["missing", "mismatched"])
def test_authority_only_bundle_cli_rejects_invalid_schedule_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    invalid_context: str,
) -> None:
    _repo, revision = _authority_repo(tmp_path)
    expected = {
        "schedule_origin_step": 12_000,
        "current_step": 12_000,
        "optimizer_count_at_current_step": 12_000,
    }
    observed = None if invalid_context == "missing" else {
        "schedule_origin_step": 0,
        "current_step": 0,
        "optimizer_count_at_current_step": 0,
    }
    _request, _registry, bundle = _matrix_case(
        tmp_path,
        revision=revision,
        execution_payload=_scheduled_training_payload(
            resume_context=expected,
            optimizer_build_context=observed,
        ),
        review_required=True,
    )
    bundle_path = tmp_path / "assembled-bundle.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        orchestrate,
        "build_training_run_matrix_authority",
        lambda *_args, **_kwargs: pytest.fail("invalid schedule reached authority emission"),
    )

    assert orchestrate.main(
        [
            "preflight",
            "--authority-only",
            "--run-set-id",
            bundle.run_set_id,
            "--bundle",
            str(bundle_path),
            "--bundle-sha256",
            canonical_run_bundle_sha256(bundle),
        ]
    ) == orchestrate.EXIT_PREFLIGHT


@pytest.mark.parametrize("tamper", ["bundle", "matrix", "capsule", "run-set"])
def test_content_pinned_bundle_authority_tampering_fails_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tamper: str,
) -> None:
    _repo, revision = _authority_repo(tmp_path)
    _request, _registry, bundle = _matrix_case(tmp_path, revision=revision)
    bundle_path = tmp_path / "assembled-bundle.json"
    bundle_path.write_text(bundle.model_dump_json(indent=2) + "\n", encoding="utf-8")
    bundle_sha256 = canonical_run_bundle_sha256(bundle)
    run_set_id = bundle.run_set_id
    if tamper == "bundle":
        payload = json.loads(bundle_path.read_text(encoding="utf-8"))
        payload["resolved_inputs"] = [{"counterfeit": True}]
        bundle_path.write_text(json.dumps(payload), encoding="utf-8")
    elif tamper == "matrix":
        Path(bundle.rows[0].execution.authored_intent.uri).write_text(
            '{"counterfeit": true}\n', encoding="utf-8"
        )
    elif tamper == "capsule":
        Path(bundle.rows[0].execution.execution_capsule.uri).write_text(
            '{"counterfeit": true}\n', encoding="utf-8"
        )
    else:
        run_set_id = "counterfeit-run-set"

    result = orchestrate.main(
        [
            "preflight",
            "--authority-only",
            "--run-set-id",
            run_set_id,
            "--bundle",
            str(bundle_path),
            "--bundle-sha256",
            bundle_sha256,
        ]
    )
    assert result != 0
    assert "counterfeit" not in capsys.readouterr().out


@pytest.mark.parametrize("invalid", ["missing-policy", "synthetic", "dirty", "missing-ref", "raw-commit-ref", "intent", "row-identity", "missing-row", "reordered-rows"])
def test_invalid_matrix_authority_stops_before_transport(tmp_path: Path, invalid: str) -> None:
    repo, revision = _authority_repo(tmp_path)
    bundle = _matrix_bundle(
        tmp_path,
        revision="a" * 40 if invalid == "synthetic" else revision,
    )
    transport = RecordingTransport()
    driver = _driver(bundle, repo, transport)
    if invalid in {"missing-policy", "missing-ref", "raw-commit-ref"}:
        refs = {
            "missing-ref": "refs/heads/not-protected",
            "raw-commit-ref": revision,
        }
        protected = {} if invalid == "missing-policy" else {"science": refs[invalid]}
        driver.config = RunPodDriverConfig(
            image=bundle.environment.image_id or "",
            local_repos={"science": repo},
            remote_repos={"science": "/workspace/science"},
            primary_repo="science",
            protected_refs=protected,
        )
    elif invalid == "dirty":
        (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    elif invalid == "intent":
        rows = []
        for row in bundle.rows:
            authored = row.execution.authored_intent.model_copy(update={"intent_hash": "a" * 64})
            execution = row.execution.model_copy(update={"authored_intent": authored})
            rows.append(row.model_copy(update={"execution": execution}))
        bundle = bundle.model_copy(update={"rows": rows})
        driver = _driver(bundle, repo, transport)
    elif invalid == "row-identity":
        row = bundle.rows[0]
        provenance = row.execution.row_provenance.model_copy(update={"planned_run_id": "fake"})
        execution = row.execution.model_copy(update={"row_provenance": provenance})
        bundle = bundle.model_copy(update={"rows": [row.model_copy(update={"execution": execution}), *bundle.rows[1:]]})
        driver = _driver(bundle, repo, transport)
    elif invalid in {"missing-row", "reordered-rows"}:
        rows = bundle.rows[:-1] if invalid == "missing-row" else list(reversed(bundle.rows))
        bundle = bundle.model_copy(update={"rows": rows})
        driver = _driver(bundle, repo, transport)

    checks = driver.preflight_checks(bundle)
    assert [(check.name, check.status) for check in checks] == [("training-matrix-authority", "fail")]
    assert transport.operations == []


@pytest.mark.parametrize("tamper", ["missing", "v1-only", "row", "monitor", "digest", "stale-code"])
def test_matrix_restore_rejects_missing_legacy_or_tampered_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    repo, revision = _authority_repo(tmp_path)
    bundle = _matrix_bundle(tmp_path, revision=revision)
    state = _completed_preflight(bundle, repo, tmp_path)
    preflight = state.stage(STAGE_PREFLIGHT)
    evidence = json.loads(json.dumps(preflight.outputs["driver_evidence"]))
    if tamper == "stale-code":
        (repo / "authority.txt").write_text("new protected authority\n", encoding="utf-8")
        _git(repo, "add", "authority.txt")
        _git(repo, "commit", "-m", "advance protected authority")
    if tamper == "missing":
        outputs = {"checks": preflight.outputs["checks"]}
    elif tamper == "v1-only":
        outputs = {**preflight.outputs, "driver_evidence": evidence["v1"]}
    else:
        if tamper == "row":
            evidence["matrix_binding"]["rows"].pop()
        elif tamper == "monitor":
            evidence["matrix_binding"]["monitor"]["event_paths"].pop()
        elif tamper == "digest":
            evidence["matrix_binding_sha256"] = "0" * 64
        outputs = {**preflight.outputs, "driver_evidence": evidence}
    state = state.with_stage(STAGE_PREFLIGHT, preflight.model_copy(update={"outputs": outputs}))
    transport = RecordingTransport()
    with pytest.raises(RunPodDriverError, match="matrix PREFLIGHT|invalid shape|code authority"):
        _driver(bundle, repo, transport).restore_completed_preflight(bundle, state)
    assert transport.operations == []

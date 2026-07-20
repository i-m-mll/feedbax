from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import pytest

from feedbax.contracts.run_matrix import (
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
    TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    TrainingRowLoweringResult,
    TrainingRunMatrixSpec,
)
from feedbax.contracts.spec_storage import training_spec_canonical_bytes
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
)
from feedbax.orchestration.drivers.runpod import (
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION,
    RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2,
    RunPodDriverConfig,
    RunPodDriverError,
    RunPodOrchestrationDriver,
)
from feedbax.orchestration.stages import STAGE_PREFLIGHT, StageEngine
from feedbax.orchestration.state import RunSetState, RunSetStateStore
from feedbax.training.spec_storage import (
    TRAINING_RUN_MATRIX_COMPILER_ID,
    TRAINING_RUN_MATRIX_COMPILER_VERSION,
    register_training_run_matrix_compiler,
)


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
        env={**os.environ, "GIT_OPTIONAL_LOCKS": "0"},
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
    _git(repo, "add", "authority.txt")
    _git(repo, "commit", "-m", "protected authority")
    return repo, _git(repo, "rev-parse", "HEAD")


def _matrix_bundle(root: Path, *, revision: str) -> RunBundle:
    matrix = TrainingRunMatrixSpec.model_validate(
        {
            "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
            "schema_version": TRAINING_RUN_MATRIX_SPEC_SCHEMA_VERSION,
            "name": "authenticated matrix",
            "base": {"kind": "inline", "inline": {"gain": 2}},
            "rows": [
                {"row_id": "first", "overrides": []},
                {"row_id": "second", "overrides": []},
            ],
        }
    )
    authored = matrix.model_dump(mode="json", exclude_none=True)
    authored_bytes = training_spec_canonical_bytes(authored)
    authored_path = root / "matrix.json"
    authored_path.write_bytes(authored_bytes)
    lockfile = root / "uv.lock"
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
            review_required=False,
            review_authorized=False,
            resources=DeploymentResourceRequest(gpu_id="RTX", regions=["CA"]),
        ),
        environment=EnvironmentDeclaration(
            python_version="3.12",
            repo_revisions=[RepoRevision(path="science", revision=revision, dirty_allowed=False)],
            lockfile_hashes={"uv.lock": hashlib.sha256(lockfile.read_bytes()).hexdigest()},
            image_id="example.invalid/feedbax@sha256:" + "a" * 64,
        ),
        budget=BudgetPolicy(max_wall_clock_seconds=30),
        orchestration_root=str(root / "runs"),
    )
    registry = AssemblyCompilerRegistry()

    def lower(authored_row: Any) -> TrainingRowLoweringResult:
        return TrainingRowLoweringResult(
            execution_payload={
                "schema_id": "example.training_payload",
                "schema_version": "example.training_payload.v1",
                "training_config": {"n_batches": 7},
                "gain": authored_row.payload["gain"],
            },
            lowerer_identities=[
                {"lowerer_id": "tests.matrix", "lowerer_version": "tests.matrix.v1"}
            ],
        )

    register_training_run_matrix_compiler(
        registry,
        allow_inline_base=True,
        row_lowerer=lower,
    )
    return assemble_run_bundle(
        request,
        run_set_id="matrix-run-set",
        context=AssemblyContext(custody_root=root / "custody", repo_root=root),
        registry=registry,
    )


def _driver(bundle: RunBundle, repo: Path, transport: RecordingTransport):
    return RunPodOrchestrationDriver(
        config=RunPodDriverConfig(
            ssh_host="203.0.113.10",
            ssh_port=22,
            image=bundle.environment.image_id or "",
            local_repos={"science": repo},
            protected_refs={"science": "refs/heads/main"},
        ),
        transport=transport,
    )


def _completed_preflight(
    bundle: RunBundle,
    repo: Path,
    root: Path,
) -> tuple[RunSetState, RunSetStateStore]:
    store = RunSetStateStore(root / "state.json")
    state = StageEngine(
        bundle=bundle,
        driver=_driver(bundle, repo, RecordingTransport()),
        store=store,
    ).run(stop_after_stage=STAGE_PREFLIGHT)
    return state, store


def test_matrix_preflight_emits_canonical_v2_without_private_paths(tmp_path: Path) -> None:
    repo, revision = _authority_repo(tmp_path)
    bundle = _matrix_bundle(tmp_path, revision=revision)
    state, _ = _completed_preflight(bundle, repo, tmp_path)
    evidence = state.stage(STAGE_PREFLIGHT).outputs["driver_evidence"]

    assert evidence["schema_version"] == RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION_V2
    assert evidence["v1"]["schema_version"] == RUNPOD_PREFLIGHT_EVIDENCE_SCHEMA_VERSION
    binding = evidence["matrix_binding"]
    assert [row["row_id"] for row in binding["rows"]] == ["first", "second"]
    assert [row["locked_training_depth"] for row in binding["rows"]] == [7, 7]
    assert binding["monitor"]["event_paths"] == [
        "events/first.events.jsonl",
        "events/second.events.jsonl",
    ]
    assert binding["code_authorities"] == [
        {
            "repo": "science",
            "declared_revision": revision,
            "protected_ref": "refs/heads/main",
            "protected_revision": revision,
            "observed_revision": revision,
            "clean": True,
        }
    ]
    assert str(tmp_path) not in json.dumps(evidence)


@pytest.mark.parametrize(
    "invalid",
    ["missing-policy", "synthetic", "dirty", "missing-ref", "raw-commit-ref", "intent"],
)
def test_invalid_matrix_authority_stops_before_transport(tmp_path: Path, invalid: str) -> None:
    repo, revision = _authority_repo(tmp_path)
    bundle = _matrix_bundle(
        tmp_path,
        revision="a" * 40 if invalid == "synthetic" else revision,
    )
    transport = RecordingTransport()
    driver = _driver(bundle, repo, transport)
    if invalid == "missing-policy":
        driver.config = RunPodDriverConfig(
            image=bundle.environment.image_id or "",
            local_repos={"science": repo},
        )
    elif invalid == "dirty":
        (repo / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    elif invalid == "missing-ref":
        driver.config = RunPodDriverConfig(
            image=bundle.environment.image_id or "",
            local_repos={"science": repo},
            protected_refs={"science": "refs/heads/not-protected"},
        )
    elif invalid == "raw-commit-ref":
        driver.config = RunPodDriverConfig(
            image=bundle.environment.image_id or "",
            local_repos={"science": repo},
            protected_refs={"science": revision},
        )
    elif invalid == "intent":
        rows = []
        for row in bundle.rows:
            authored = row.execution.authored_intent.model_copy(
                update={"intent_hash": "a" * 64}
            )
            execution = row.execution.model_copy(update={"authored_intent": authored})
            rows.append(row.model_copy(update={"execution": execution}))
        bundle = bundle.model_copy(update={"rows": rows})
        driver = _driver(bundle, repo, transport)

    checks = driver.preflight_checks(bundle)

    assert [(check.name, check.status) for check in checks] == [
        ("training-matrix-authority", "fail")
    ]
    assert transport.operations == []


@pytest.mark.parametrize(
    "tamper", ["missing", "v1-only", "row", "monitor", "digest", "stale-code"]
)
def test_matrix_restore_rejects_missing_legacy_or_tampered_evidence(
    tmp_path: Path,
    tamper: str,
) -> None:
    repo, revision = _authority_repo(tmp_path)
    bundle = _matrix_bundle(tmp_path, revision=revision)
    state, _ = _completed_preflight(bundle, repo, tmp_path)
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
    state = state.with_stage(
        STAGE_PREFLIGHT,
        preflight.model_copy(update={"outputs": outputs}),
    )
    transport = RecordingTransport()

    with pytest.raises(
        RunPodDriverError, match="matrix PREFLIGHT|invalid shape|code authority"
    ):
        _driver(bundle, repo, transport).restore_completed_preflight(bundle, state)

    assert transport.operations == []

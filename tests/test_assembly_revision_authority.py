"""Tests for the assembly-time Feedbax revision authority (Mandible-Issue a09bad3).

``RunBundle.feedbax_revision`` used to be minted from whichever package happened
to supply ``import feedbax``, and preflight then compared that bundle against the
same imported package. A stale editable install was therefore self-consistent and
passed its own gate. ``RunAssemblyRequest.v7`` carries the revision authority
instead, it is verified against real imported-package provenance before anything
is compiled or written, and the authored value is copied into the bundle.

Every test here points ``revision.feedbax.__file__`` at an isolated ``tmp_path``
Git checkout rather than the real package source, so the checks run against real
Git provenance, are pytest-xdist-safe, and leave no ambient state behind.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import pytest

from feedbax.contracts.migrations import UnsupportedSpecVersion, default_spec_registry
from feedbax.contracts.spec_storage import training_spec_canonical_bytes
from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingIdentityAdapter,
)
from feedbax.orchestration import revision
from feedbax.orchestration.assembly import (
    RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION,
    RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V6,
    AssemblyCompilerRegistry,
    AssemblyContext,
    CompiledExecutionRow,
    CompiledRunSet,
    CompilerIdentity,
    RunAssemblyRequest,
    assemble_run_bundle,
)
from feedbax.orchestration.bundle import (
    BudgetPolicy,
    DeploymentPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    RowLaunchSpec,
    SchemaArtifactRef,
)
from feedbax.orchestration.revision import (
    FeedbaxRevisionError,
    assert_feedbax_source_residence,
)


FABRICATED_REVISION = "0123456789abcdef0123456789abcdef01234567"
_COMPILER_ID = "feedbax.tests.revision-authority-fixture"
_COMPILER_VERSION = "feedbax.tests.revision-authority-fixture.v1"


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        env={**revision._GIT_ENVIRONMENT, "HOME": str(root)},
    )


def _init_clean_repo(tmp_path: Path) -> tuple[Path, str]:
    """Create a tmp Git checkout with one committed package file and return its HEAD."""
    package_root = tmp_path / "checkout" / "feedbax"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("# stub package\n")
    repo_root = package_root.parent
    _git(repo_root, "init", "--quiet")
    _git(repo_root, "config", "user.email", "test@example.com")
    _git(repo_root, "config", "user.name", "Test")
    _git(repo_root, "add", "-A")
    _git(repo_root, "commit", "--quiet", "-m", "initial")
    head = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        env={**revision._GIT_ENVIRONMENT, "HOME": str(repo_root)},
    ).stdout.strip()
    return package_root, head


@pytest.fixture
def clean_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str]:
    """Make the imported package resolve to an isolated, clean tmp Git checkout."""
    package_root, head = _init_clean_repo(tmp_path)
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))
    return package_root, head


@dataclasses.dataclass(frozen=True)
class _FixtureCompiler:
    """Minimal compiler that records whether the gate let compilation happen."""

    compiled: list[str]

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del authored, context
        self.compiled.append(run_set_id)
        return CompiledRunSet(
            rows=[
                CompiledExecutionRow(
                    row_id="row-a",
                    payload={
                        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
                        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
                        "total_batches": 1,
                        "training_config": {},
                    },
                    resolved_semantics={
                        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
                        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
                        "total_batches": 1,
                        "training_config": {},
                    },
                    launch=RowLaunchSpec(command=[sys.executable, "-c", "pass"], collect=[]),
                )
            ]
        )


@dataclasses.dataclass(frozen=True)
class _AssemblyFixture:
    """One authored request plus everything ASSEMBLE needs and writes."""

    request: RunAssemblyRequest
    context: AssemblyContext
    registry: AssemblyCompilerRegistry
    compiled: list[str]
    custody_root: Path
    orchestration_root: Path

    def output_paths(self) -> list[Path]:
        """Return every path ASSEMBLE would have created under its own roots."""
        roots = (self.custody_root, self.orchestration_root)
        return sorted(
            path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()
        )


def _assembly_fixture(tmp_path: Path, *, feedbax_revision: str) -> _AssemblyFixture:
    authored = {
        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
        "total_batches": 1,
    }
    authored_bytes = training_spec_canonical_bytes(authored)
    authored_path = tmp_path / "authored" / "authored.json"
    authored_path.parent.mkdir(parents=True, exist_ok=True)
    authored_path.write_bytes(authored_bytes)
    orchestration_root = tmp_path / "orchestration"
    custody_root = tmp_path / "custody"
    request = RunAssemblyRequest(
        feedbax_revision=feedbax_revision,
        authored=SchemaArtifactRef(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            artifact_id="fixture:authored",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler=CompilerIdentity(
            compiler_id=_COMPILER_ID,
            compiler_version=_COMPILER_VERSION,
        ),
        deployment_policy=DeploymentPolicy(
            driver="local",
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(python_version="3.12"),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=10.0),
        orchestration_root=str(orchestration_root),
    )
    compiled: list[str] = []
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=_COMPILER_ID,
        compiler_version=_COMPILER_VERSION,
        compiler=_FixtureCompiler(compiled),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    return _AssemblyFixture(
        request=request,
        context=AssemblyContext(custody_root=custody_root),
        registry=registry,
        compiled=compiled,
        custody_root=custody_root,
        orchestration_root=orchestration_root,
    )


def _assemble(fixture: _AssemblyFixture):
    return assemble_run_bundle(
        fixture.request,
        run_set_id="2026-01-02-deadbeef",
        context=fixture.context,
        registry=fixture.registry,
    )


def test_matching_clean_install_assembles_and_copies_the_authored_revision(
    tmp_path: Path, clean_checkout: tuple[Path, str]
) -> None:
    _package_root, head = clean_checkout
    fixture = _assembly_fixture(tmp_path, feedbax_revision=head)

    bundle = _assemble(fixture)

    assert bundle.feedbax_revision == head
    assert fixture.compiled == ["2026-01-02-deadbeef"]
    # The negative cases below assert this same set is empty, so it must be
    # non-empty when the gate passes for those assertions to mean anything.
    assert fixture.output_paths()


def test_wrong_authored_revision_fails_closed_before_any_output(
    tmp_path: Path, clean_checkout: tuple[Path, str]
) -> None:
    _package_root, head = clean_checkout
    fixture = _assembly_fixture(tmp_path, feedbax_revision=FABRICATED_REVISION)

    with pytest.raises(FeedbaxRevisionError) as excinfo:
        _assemble(fixture)

    message = str(excinfo.value)
    assert FABRICATED_REVISION in message
    assert head in message
    assert "feedbax_revision" in message
    # Nothing compiled, and no custody or run-set bytes were written.
    assert fixture.compiled == []
    assert fixture.output_paths() == []


def test_dirty_supplying_checkout_fails_closed(
    tmp_path: Path, clean_checkout: tuple[Path, str]
) -> None:
    """Dirty handling matches ``check_feedbax_provenance``: a dirty tree never passes."""
    package_root, head = clean_checkout
    (package_root / "__init__.py").write_text("# uncommitted edit\n")
    fixture = _assembly_fixture(tmp_path, feedbax_revision=head)

    with pytest.raises(FeedbaxRevisionError, match="uncommitted changes"):
        _assemble(fixture)

    assert fixture.compiled == []
    assert fixture.output_paths() == []


def test_unverifiable_provenance_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A package that is neither a Git checkout nor a provenance-bearing wheel rejects."""
    package_root = tmp_path / "not-a-checkout" / "feedbax"
    package_root.mkdir(parents=True)
    (package_root / "__init__.py").write_text("# stub package\n")
    monkeypatch.setattr(revision.feedbax, "__file__", str(package_root / "__init__.py"))
    fixture = _assembly_fixture(tmp_path, feedbax_revision=FABRICATED_REVISION)

    with pytest.raises(FeedbaxRevisionError, match="cannot resolve a verified revision identity"):
        _assemble(fixture)

    assert fixture.compiled == []
    assert fixture.output_paths() == []


def test_malformed_authored_revision_is_rejected_by_the_schema(tmp_path: Path) -> None:
    """The authority is a full lowercase Git commit; the schema refuses anything else."""
    with pytest.raises(ValueError, match="feedbax_revision"):
        _assembly_fixture(tmp_path, feedbax_revision="not-a-commit")


def test_source_residence_assertion_accepts_the_real_supplying_checkout(
    clean_checkout: tuple[Path, str],
) -> None:
    package_root, _head = clean_checkout

    assert assert_feedbax_source_residence(package_root.parent) == package_root.parent


def test_source_residence_assertion_refuses_a_different_checkout(
    tmp_path: Path, clean_checkout: tuple[Path, str]
) -> None:
    package_root, _head = clean_checkout
    other_checkout = tmp_path / "some" / "other" / "checkout"
    other_checkout.mkdir(parents=True)

    with pytest.raises(FeedbaxRevisionError, match="source residence assertion failed") as excinfo:
        assert_feedbax_source_residence(other_checkout)

    message = str(excinfo.value)
    assert str(other_checkout) in message
    assert str(package_root.parent) in message


def test_v6_request_is_explicitly_rejected_with_a_reauthoring_instruction(
    tmp_path: Path, clean_checkout: tuple[Path, str]
) -> None:
    """A v6 request cannot be migrated: the migrator has no revision to invent."""
    _package_root, head = clean_checkout
    payload = _assembly_fixture(tmp_path, feedbax_revision=head).request.model_dump(mode="json")
    del payload["feedbax_revision"]
    payload["schema_version"] = RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V6

    with pytest.raises(UnsupportedSpecVersion) as excinfo:
        default_spec_registry.migrate("RunAssemblyRequest", payload)

    message = str(excinfo.value)
    assert RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V6 in message
    assert RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION in message
    assert "re-author a current v7 request" in message


def test_orchestrate_cli_rejects_a_v6_assembly_request_file(
    tmp_path: Path, clean_checkout: tuple[Path, str]
) -> None:
    """The CLI load path surfaces the rejection rather than assembling anything."""
    from feedbax.bin import orchestrate

    _package_root, head = clean_checkout
    payload = _assembly_fixture(tmp_path, feedbax_revision=head).request.model_dump(mode="json")
    del payload["feedbax_revision"]
    payload["schema_version"] = RUN_ASSEMBLY_REQUEST_SCHEMA_VERSION_V6
    path = tmp_path / "request-v6.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(UnsupportedSpecVersion, match="re-author a current v7 request"):
        orchestrate._load_assembly_request(path)

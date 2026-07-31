"""Bounded public production lifecycle exercised from installed wheels."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import errno
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from typing import Any

from feedbax.contracts.studio_training import (
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
    STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
    StudioTrainingIdentityAdapter,
)
from feedbax.orchestration import (
    AssemblyCompilerRegistry,
    AssemblyContext,
    BudgetPolicy,
    CheckEntry,
    CheckRegistry,
    CompiledExecutionRow,
    CompiledRunSet,
    CustodyPreservationRequired,
    DeploymentPolicy,
    EnvironmentDeclaration,
    LaunchPolicy,
    PrimaryStatePersistenceError,
    RowLaunchSpec,
    RunAssemblyRequest,
    RunSetState,
    RunSetStateStore,
    SchemaArtifactRef,
    StageEngine,
)
from feedbax.orchestration.drivers import (
    AcquisitionSemantics,
    AuthorizationSemantics,
    CustodySemantics,
    DriverCapabilityEnvelope,
    DriverCapabilityFacts,
    DriverConstructionContext,
    DriverHook,
    DriverRegistration,
    DriverRegistry,
    DriverRowProbe,
    DriverVenue,
    EnvironmentSemantics,
    MonitoringSemantics,
    RealizedDriverCapabilities,
    RecoverySemantics,
    ResourceSemantics,
    RetrySemantics,
    SpendSemantics,
    TeardownSemantics,
)
from feedbax.orchestration.revision import resolve_feedbax_revision
from feedbax.plugins.application import new_application_registry_bundle


_COMPILER_ID = "feedbax-external-conformance.local-lifecycle"
_COMPILER_VERSION = f"{_COMPILER_ID}.v1"
_RUN_SET_ID = "external-conformance-local"
_LOCAL_LIFECYCLE_SCRIPT = "print('feedbax external conformance lifecycle')"


@dataclass(frozen=True)
class _LocalLifecycleCompiler:
    """Incubated fixture compiler for one deterministic bounded local row."""

    def compile(
        self,
        *,
        authored: Mapping[str, Any],
        run_set_id: str,
        context: AssemblyContext,
    ) -> CompiledRunSet:
        del context
        payload = dict(authored)
        return CompiledRunSet(
            rows=[
                CompiledExecutionRow(
                    row_id=f"{run_set_id}-row",
                    payload=payload,
                    resolved_semantics=payload,
                    launch=RowLaunchSpec(
                        command=[
                            sys.executable,
                            "-c",
                            _LOCAL_LIFECYCLE_SCRIPT,
                        ],
                    ),
                )
            ]
        )


def _request(
    root: Path,
    *,
    driver: str = "local",
) -> tuple[RunAssemblyRequest, AssemblyContext, AssemblyCompilerRegistry]:
    authored = {
        "schema_id": STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        "schema_version": STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
        "total_batches": 1,
        "training_config": {"fixture": "bounded-local-lifecycle-v1"},
    }
    authored_bytes = json.dumps(
        authored,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    authored_path = root / "authored.json"
    authored_path.write_bytes(authored_bytes)
    request = RunAssemblyRequest(
        authored=SchemaArtifactRef(
            schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
            schema_version=STUDIO_TRAINING_ASSEMBLY_SCHEMA_VERSION,
            artifact_id="fixture:bounded-local-lifecycle-v1",
            sha256=hashlib.sha256(authored_bytes).hexdigest(),
            uri=str(authored_path),
        ),
        compiler={
            "compiler_id": _COMPILER_ID,
            "compiler_version": _COMPILER_VERSION,
        },
        deployment_policy=DeploymentPolicy(
            driver=driver,
            venue="local",
            cloud_authorized=False,
            review_required=False,
            review_authorized=False,
        ),
        environment=EnvironmentDeclaration(
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}"
        ),
        launch_policy=LaunchPolicy(max_parallel_rows=1),
        budget=BudgetPolicy(max_wall_clock_seconds=10.0, max_spend_usd=0.0),
        orchestration_root=str(root / "orchestration"),
    )
    registry = AssemblyCompilerRegistry()
    registry.register(
        schema_id=STUDIO_TRAINING_ASSEMBLY_SCHEMA_ID,
        compiler_id=_COMPILER_ID,
        compiler_version=_COMPILER_VERSION,
        compiler=_LocalLifecycleCompiler(),
        identity_adapter=StudioTrainingIdentityAdapter(),
    )
    return request, AssemblyContext(custody_root=root / "custody"), registry


class _MonitorEnospcStore(RunSetStateStore):
    """Inject one primary-state ENOSPC after a resource has been launched."""

    def __init__(self, path: Path) -> None:
        super().__init__(path)
        self.primary_failed = False

    def save(self, state: RunSetState, *, crash_before_replace: bool = False) -> Path:
        if (
            not self.primary_failed
            and state.current_stage == "MONITOR"
            and state.stage("MONITOR").status == "running"
        ):
            self.primary_failed = True
            cause = OSError(errno.ENOSPC, "external fixture primary state ENOSPC")
            raise PrimaryStatePersistenceError(self.path, cause) from cause
        return super().save(state, crash_before_replace=crash_before_replace)


class _CustodyDriver:
    """Allocation-free destructive fixture governed by realized capability facts."""

    poll_interval_seconds = 0.01
    capability_envelope = DriverCapabilityEnvelope.single(
        "fixture:custody",
        DriverCapabilityFacts(
            variant_id="destructive-ephemeral",
            venue=DriverVenue.LOCAL_PROCESS,
            resources=ResourceSemantics.DRIVER_OWNED,
            spend=SpendSemantics.NONE,
            authorization=AuthorizationSemantics.NONE,
            environment=EnvironmentSemantics.LOCAL_INVENTORY,
            monitoring=MonitoringSemantics.ROW_POLL,
            recovery=RecoverySemantics.NONE,
            retry=RetrySemantics.NONE,
            acquisition=AcquisitionSemantics.EXTERNALLY_PROVIDED,
            teardown=TeardownSemantics.VERIFIED_RESOURCE_ABSENCE,
            custody=CustodySemantics.EPHEMERAL_REMOTE_RESOURCE,
            optional_hooks=frozenset({DriverHook.TEARDOWN_OWNERSHIP}),
        ),
    )

    def __init__(
        self,
        realized: RealizedDriverCapabilities,
        *,
        fail_collect: bool = False,
    ) -> None:
        self.realized_capabilities = realized
        self.fail_collect = fail_collect
        self.delete_calls = 0

    def provision(self, *_args: object) -> dict[str, object]:
        return {"driver": "fixture:custody", "resource_id": "fixture-resource"}

    def realize_env(self, *_args: object) -> str:
        return "fixture-custody-environment"

    def stage_inputs(self, *_args: object) -> dict[str, object]:
        return {}

    def launch_row(self, *_args: object) -> dict[str, object]:
        return {"pid": 1}

    def probe(self, *_args: object) -> DriverRowProbe:
        return DriverRowProbe(status="completed")

    def stop_row(self, *_args: object) -> dict[str, object]:
        return {}

    def collect(self, *_args: object) -> dict[str, str]:
        if self.fail_collect:
            raise RuntimeError("fixture collection interrupted before custody")
        return {}

    def teardown_ownership(self, _state: RunSetState) -> dict[str, object]:
        return {
            "kind": "fixture-created",
            "owned_by_run": True,
            "teardown_allowed": True,
            "resource_id": "fixture-resource",
        }

    def teardown(self, *_args: object) -> dict[str, object]:
        self.delete_calls += 1
        return {"teardown": "removed", "resource_id": "fixture-resource"}


def _custody_driver(*, fail_collect: bool = False) -> _CustodyDriver:
    return _CustodyDriver(
        _CustodyDriver.capability_envelope.realize("destructive-ephemeral"),
        fail_collect=fail_collect,
    )


def _custody_driver_registry(driver: _CustodyDriver) -> DriverRegistry:
    envelope = _CustodyDriver.capability_envelope
    return DriverRegistry(
        (
            DriverRegistration(
                name="fixture:custody",
                supported_capabilities=envelope,
                resolve_capabilities=lambda _context: driver.realized_capabilities,
                factory=lambda _context, _realized: driver,
            ),
        )
    )


def _driver_context(root: Path, bundle: object) -> DriverConstructionContext:
    return DriverConstructionContext(
        configuration={"bundle": bundle},
        runtime_bindings={
            "cwd": str(root),
            "python_executable": sys.executable,
        },
    )


def check_public_lifecycle_recovery() -> bool:
    """Run and resume the installed-wheel local production lifecycle."""
    with tempfile.TemporaryDirectory(prefix="feedbax-external-conformance-") as temporary:
        root = Path(temporary).resolve()
        request, context, registry = _request(root)
        store = RunSetStateStore(root / "orchestration" / _RUN_SET_ID / "state.json")
        checks = CheckRegistry(
            {
                "external_fixture": lambda _row: CheckEntry(
                    check_id="external_fixture",
                    status="pass",
                )
            }
        )
        registries = new_application_registry_bundle(local_component_source=None)
        initial_engine = StageEngine.from_request(
            request,
            context=context,
            registry=registry,
            driver_registry=registries.drivers,
            driver_context=lambda bundle: _driver_context(root, bundle),
            run_set_id=_RUN_SET_ID,
            store=store,
            conformance_registry=checks,
        )
        first = initial_engine.run(stop_after_stage="PREFLIGHT")
        assert initial_engine.bundle is not None
        expected_command = [sys.executable, "-c", _LOCAL_LIFECYCLE_SCRIPT]
        observed_rows = [(row.row_id, row.launch.command) for row in initial_engine.bundle.rows]
        expected_rows = [(f"{_RUN_SET_ID}-row", expected_command)]
        if observed_rows != expected_rows:
            raise AssertionError(
                "bounded lifecycle row inventory drifted from the single print-only child"
            )
        revision_check = next(
            check
            for check in first.stage("PREFLIGHT").checks
            if check.name == "feedbax-revision-pin"
        )
        installed_revision = resolve_feedbax_revision()
        if revision_check.status != "pass" or revision_check.observed != installed_revision:
            raise AssertionError("installed-wheel revision gate did not authenticate PREFLIGHT")
        persisted = store.load()
        if persisted.stage("ASSEMBLE").attempts != 1 or persisted.stage("PREFLIGHT").attempts != 1:
            raise AssertionError(
                "public lifecycle state was not persisted at the recovery boundary"
            )

        recovered = StageEngine.from_request(
            request,
            context=context,
            registry=registry,
            driver_registry=registries.drivers,
            driver_context=lambda bundle: _driver_context(root, bundle),
            run_set_id=_RUN_SET_ID,
            store=store,
            conformance_registry=checks,
        ).run()
        if recovered.stage("ASSEMBLE").attempts != 1:
            raise AssertionError("recovery reran completed ASSEMBLE state")
        if recovered.stage("PREFLIGHT").attempts != 1:
            raise AssertionError("recovery reran completed PREFLIGHT state")
        if recovered.stage("LAUNCH").status != "completed":
            raise AssertionError("recovered lifecycle did not execute LAUNCH")
        if recovered.stage("REGISTER").status != "completed":
            raise AssertionError("recovered lifecycle did not finish registration")
        if recovered.rows[f"{_RUN_SET_ID}-row"].status != "completed":
            raise AssertionError("bounded local lifecycle row did not complete")
        if store.load() != recovered:
            raise AssertionError("recovered lifecycle result was not persisted")
    return True


def check_custody_persistence_recovery() -> bool:
    """Prove primary ENOSPC survives restart and gates deletion until custody."""
    with tempfile.TemporaryDirectory(prefix="feedbax-custody-conformance-") as temporary:
        root = Path(temporary).resolve()
        request, context, registry = _request(root, driver="fixture:custody")
        state_path = root / "orchestration" / _RUN_SET_ID / "state.json"
        store = _MonitorEnospcStore(state_path)
        checks = CheckRegistry(
            {
                "external_fixture": lambda _row: CheckEntry(
                    check_id="external_fixture",
                    status="pass",
                )
            }
        )

        initial_driver = _custody_driver()
        try:
            StageEngine.from_request(
                request,
                context=context,
                registry=registry,
                driver_registry=_custody_driver_registry(initial_driver),
                driver_context=lambda bundle: _driver_context(root, bundle),
                run_set_id=_RUN_SET_ID,
                store=store,
                conformance_registry=checks,
            ).run()
        except PrimaryStatePersistenceError as exc:
            if exc.cause.errno != errno.ENOSPC:
                raise AssertionError("primary fault did not retain ENOSPC identity") from exc
        else:
            raise AssertionError("primary ENOSPC did not interrupt the installed lifecycle")

        emergency = store.load_emergency()
        if (
            emergency.provider_identity.provider != "fixture:custody"
            or emergency.provider_identity.resource_id != "fixture-resource"
            or emergency.preservation_state != "preserve-required"
            or emergency.custody_complete
        ):
            raise AssertionError("primary ENOSPC emergency evidence was incomplete")
        if initial_driver.delete_calls:
            raise AssertionError("primary persistence failure reached destructive teardown")

        restarted_store = RunSetStateStore(state_path)
        restarted_store.preflight_and_reserve()
        blocked_driver = _custody_driver(fail_collect=True)
        try:
            StageEngine.from_request(
                request,
                context=context,
                registry=registry,
                driver_registry=_custody_driver_registry(blocked_driver),
                driver_context=lambda bundle: _driver_context(root, bundle),
                run_set_id=_RUN_SET_ID,
                store=restarted_store,
                conformance_registry=checks,
            ).run()
        except CustodyPreservationRequired as exc:
            if "requires custody first" not in str(exc):
                raise AssertionError("custody gate reported an unexpected reason") from exc
        else:
            raise AssertionError("restart deletion was not blocked before custody")
        if blocked_driver.delete_calls:
            raise AssertionError("restart invoked delete before custody")
        if restarted_store.load_emergency().preservation_state != "preserve-required":
            raise AssertionError("blocked restart weakened emergency preservation state")

        restarted_store.preflight_and_reserve()
        release_driver = _custody_driver()
        released = StageEngine.from_request(
            request,
            context=context,
            registry=registry,
            driver_registry=_custody_driver_registry(release_driver),
            driver_context=lambda bundle: _driver_context(root, bundle),
            run_set_id=_RUN_SET_ID,
            store=restarted_store,
            conformance_registry=checks,
        ).run(stop_after_stage="TEARDOWN")
        row_id = f"{_RUN_SET_ID}-row"
        if released.rows[row_id].status != "completed":
            raise AssertionError("restarted row did not reach collected custody")
        if released.stage("COLLECT").status != "completed":
            raise AssertionError("restarted lifecycle did not persist completed custody")
        if release_driver.delete_calls != 1:
            raise AssertionError("destructive teardown did not occur exactly once after custody")
        released_emergency = restarted_store.load_emergency()
        if (
            released_emergency.preservation_state != "release-authorized"
            or not released_emergency.custody_complete
        ):
            raise AssertionError("post-custody emergency evidence did not authorize release")
    return True


__all__ = ["check_custody_persistence_recovery", "check_public_lifecycle_recovery"]

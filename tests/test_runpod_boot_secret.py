from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

import httpx
import pytest

from feedbax.orchestration.bundle import RunPodCudaPlacement
from feedbax.orchestration.drivers import runpod
from feedbax.orchestration.stages import OrchestrationStageError, StageEngine
from feedbax.orchestration.state import RunSetStateStore, StageState
from tests.test_runpod_orchestration_driver import _bundle, _state


def test_boot_command_preserves_only_the_matching_key(tmp_path: Path) -> None:
    private_key, public_key = runpod._fresh_boot_host_key()
    startup = tmp_path / "start"
    startup.write_text("#!/bin/bash\nexec /usr/bin/env\n")
    startup.chmod(0o700)
    command = [
        "/bin/bash", "-c", runpod._BOOT_HOST_KEY_COMMAND, "feedbax-host-key",
        public_key, str(tmp_path), str(startup),
    ]
    for key in ("", "not base64", runpod._fresh_boot_host_key()[0]):
        failed = subprocess.run(
            command, env={"PATH": os.defpath, runpod._BOOT_HOST_KEY_ENV: key},
            capture_output=True, text=True, timeout=5,
        )
        assert failed.returncode != 0
        assert not (tmp_path / "ssh_host_ed25519_key").exists()
        assert not list(tmp_path.glob(".feedbax-host.*"))
    booted = subprocess.run(
        command, env={"PATH": os.defpath, runpod._BOOT_HOST_KEY_ENV: private_key},
        capture_output=True, text=True, timeout=5, check=True,
    )
    assert runpod._BOOT_HOST_KEY_ENV not in booted.stdout
    assert private_key not in booted.stdout + booted.stderr
    assert (tmp_path / "ssh_host_ed25519_key").stat().st_mode & 0o777 == 0o600
    assert (tmp_path / "ssh_host_ed25519_key.pub").read_text().strip() == public_key


@pytest.mark.parametrize("scenario", [
    "success", "wrong-key", "startup-timeout", "pod-rejected", "pod-ambiguous", "secret-rejected",
    "pod-ambiguous-inventory-unavailable", "pod-ambiguous-delete-unavailable",
    "secret-ambiguous", "restart-found", "restart-zero", "restart-lost-delete",
])
def test_boot_secret_is_owned_by_the_production_acquisition_intent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str,
) -> None:
    bundle = _bundle(tmp_path)
    bundle = bundle.model_copy(update={"budget": bundle.budget.model_copy(update={
        "max_create_attempts": 1,
        "teardown_reserve_seconds": 20,
        "discard_uncollected_on_abort": True,
    })})
    gpu = "NVIDIA GeForce RTX 5090"
    config = runpod.RunPodDriverConfig(
        gpu_id=gpu, datacenters=("EU-RO-1",), image=bundle.environment.image_id,
        api_key="mock-api-key", boot_known_hosts_path=bundle.run_set_dir / "ssh-known-hosts",
        cuda_placement=RunPodCudaPlacement(
            cuda_version="13.2", min_driver_major=595, compute_capability="12.0",
            ephemeral_disk_gb=20, max_hourly_rate_usd=1.19,
            startup_seconds=1 if scenario == "startup-timeout" else 7,
            ssh_host_trust="boot-secret",
        ),
    )
    private_key, public_key = runpod._fresh_boot_host_key()
    monkeypatch.setattr(runpod, "_fresh_boot_host_key", lambda: (private_key, public_key))
    store = RunSetStateStore(bundle.run_set_dir / "state.json")
    effects: list[tuple[str, str]] = []
    secret = None
    pod = None
    provider_recovered = False
    secret_path = "/v2/account/secrets"
    pod_path = "/v2/pods"

    def provider(request: httpx.Request) -> httpx.Response:
        nonlocal secret, pod
        method, path = request.method, request.url.path
        effects.append((method, path))
        if path.startswith("/v2/catalog/gpus/"):
            return httpx.Response(200, json={
                "id": gpu, "price": {"secure": 1.19},
                "cudaVersions": [{"version": "13.2", "available": True}],
                "dataCenters": [{"id": "EU-RO-1", "availability": "HIGH"}],
            })
        if method == "POST" and path == secret_path:
            intent, = store.load().acquisition_intents
            assert intent.preparation and intent.preparation.provider_id is None
            payload = json.loads(request.content)
            assert payload.pop("value") == private_key
            assert payload == runpod._boot_secret_metadata(intent.intent_id)
            if scenario == "secret-rejected":
                return httpx.Response(403)
            secret = {**payload, "id": "secret-one"}
            if scenario == "secret-ambiguous":
                raise httpx.ReadTimeout("secret response lost")
            return httpx.Response(201, json=secret)
        if method == "POST" and path == pod_path:
            intent, = store.load().acquisition_intents
            assert intent.preparation and intent.preparation.provider_id == "secret-one"
            assert intent.preparation.public_identity == public_key
            assert private_key.encode() not in request.content
            body = json.loads(request.content)
            assert body["env"] == {runpod._BOOT_HOST_KEY_ENV: "{{ RUNPOD_SECRET_" + secret["name"] + " }}"}
            assert body["entrypoint"][4] == public_key
            if scenario == "pod-rejected":
                return httpx.Response(
                    400, headers={"Content-Type": "application/problem+json"},
                    json={"title": "Bad Request", "status": 400, "detail": "no capacity"},
                )
            pod = {
                **body, "id": "pod-one", "status": "RUNNING", "mounts": {},
                "dataCenterId": "EU-RO-1", "cudaVersion": "13.2", "cost": 1.19,
                "createdAt": datetime.now(timezone.utc).isoformat(),
                "ssh": {"direct": {"host": "203.0.113.10", "port": 2222, "username": "root"}},
            }
            if scenario.startswith("pod-ambiguous"):
                raise httpx.ReadTimeout("reply lost")
            return httpx.Response(201, json={"id": "pod-one"})
        if path == secret_path:
            if secret is not None:
                assert request.url.params["name"] == secret["name"]
            return httpx.Response(200, json={"secrets": [secret] if secret else []})
        if path == f"{secret_path}/secret-one":
            if method == "DELETE":
                assert pod is None
                intent, = store.load().acquisition_intents
                assert intent.preparation.provider_id == "secret-one"
                if intent.preparation.public_identity is not None:
                    assert intent.state in {"failed-unacquired", "resolved-torn-down"}
                secret = None
                if scenario == "restart-lost-delete":
                    raise httpx.ReadTimeout("delete response lost")
                return httpx.Response(204)
            return httpx.Response(200, json=secret) if secret else httpx.Response(404)
        if path == f"{pod_path}/pod-one":
            if method == "DELETE":
                if scenario == "pod-ambiguous-delete-unavailable" and not provider_recovered:
                    return httpx.Response(503)
                pod = None
                return httpx.Response(204)
            return httpx.Response(200, json=pod) if pod else httpx.Response(404)
        if path == pod_path:
            if scenario == "pod-ambiguous-inventory-unavailable" and not provider_recovered:
                return httpx.Response(503)
            return httpx.Response(200, json={
                "pods": [{"id": pod["id"], "name": pod["name"]}] if pod else [],
                "pagination": {"hasNextPage": False},
            })
        assert path == "/v2/billing/pods"
        return httpx.Response(200, json={"records": []})

    def local_ssh(args, **kwargs):
        assert args[0] == "ssh"
        assert "StrictHostKeyChecking=yes" in args
        assert config.boot_known_hosts_path.read_text() == f"[203.0.113.10]:2222 {public_key}\n"
        if scenario == "wrong-key":
            return runpod.CommandResult(255, "", "REMOTE HOST IDENTIFICATION HAS CHANGED")
        if scenario == "startup-timeout":
            return runpod.CommandResult(255, "", "Connection refused")
        if args[-1].startswith("nvidia-smi"):
            return runpod.CommandResult(
                0, f"{gpu}, 595.2, 12.0, GPU-00000000-0000-0000-0000-000000000000",
            )
        return runpod.CommandResult(0, "")

    monkeypatch.setattr(runpod, "_run_command", local_ssh)

    def new_engine() -> StageEngine:
        driver = runpod.RunPodOrchestrationDriver(
            config=config, transport=runpod.SubprocessRunPodTransport(
                http_transport=httpx.MockTransport(provider),
            ),
        )
        driver._preflight_passed = True
        engine = StageEngine(bundle=bundle, driver=driver, store=store)
        monkeypatch.setattr(engine, "_restore_completed_driver_preflight", lambda state: state)
        return engine

    state = _state(bundle)
    for stage in ("ASSEMBLE", "PREFLIGHT"):
        state = state.with_stage(stage, StageState(status="completed"))
    store.save(state)
    engine = new_engine()
    if scenario.startswith("restart"):
        _, intent = engine._new_acquisition_intent(
            state, attempt_ordinal=1, candidate_ordinal=1, candidate="EU-RO-1",
        )
        if scenario == "restart-zero":
            with pytest.raises(runpod.RunPodDriverError, match="reconcile.*by name"):
                new_engine().abort()
            persisted, = store.load().acquisition_intents
            assert persisted.state == "intended" and not persisted.preparation.released
            assert effects == [("GET", secret_path)]
        secret = {**runpod._boot_secret_metadata(intent.intent_id), "id": "secret-one"}
        if scenario == "restart-lost-delete":
            with pytest.raises(runpod.RunPodDriverError, match="transport failed"):
                new_engine().abort()
            persisted, = store.load().acquisition_intents
            assert persisted.preparation.provider_id == "secret-one"
            assert not persisted.preparation.released
        assert new_engine().abort().stage("TEARDOWN").status == "completed"
        assert ("POST", pod_path) not in effects
        assert ("GET", pod_path) not in effects
    elif scenario == "success":
        live = engine.run(stop_after_stage="PROVISION")
        assert live.provision_record["ssh_boot_public_key"] == public_key
        assert secret is not None
        config.boot_known_hosts_path.unlink()
        resumed = new_engine()
        resumed._restore_driver_from_provision_record(live)
        resumed.driver._ssh("true")
        assert resumed.abort().stage("TEARDOWN").status == "completed"
    elif scenario.endswith("-unavailable"):
        with pytest.raises(runpod.RunPodDriverError, match="inventory response was invalid|teardown is unresolved"):
            engine.run(stop_after_stage="PROVISION")
        persisted, = store.load().acquisition_intents
        assert persisted.state in {"ambiguous", "ambiguous-unresolved"}
        assert not persisted.preparation.released
        assert secret is not None and pod is not None
        assert ("DELETE", f"{secret_path}/secret-one") not in effects
        provider_recovered = True
        assert new_engine().abort().stage("TEARDOWN").status == "completed"
    else:
        with pytest.raises(OrchestrationStageError, match="provisioning stopped"):
            engine.run(stop_after_stage="PROVISION")
    final, = store.load().acquisition_intents
    assert final.preparation.released
    assert secret is None and pod is None
    assert private_key not in store.path.read_text()
    assert effects.count(("POST", pod_path)) == (
        0 if scenario.startswith("restart") or scenario.startswith("secret") else 1
    )

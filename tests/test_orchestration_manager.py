import asyncio

from feedbax.web.orchestration.gcp import InstanceConfig, InstanceInfo, InstanceStatus
from feedbax.web.orchestration.manager import OrchestrationManager


class _TrainingService:
    def connect_remote(self, worker_url, auth_token):
        self.worker_url = worker_url
        self.auth_token = auth_token

    def _terminate_worker(self):
        pass


def test_launch_failure_after_create_deletes_instance(monkeypatch):
    async def _run():
        manager = OrchestrationManager()
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


def test_launch_cleanup_failure_records_orphaned_instance(monkeypatch):
    async def _run():
        manager = OrchestrationManager()
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

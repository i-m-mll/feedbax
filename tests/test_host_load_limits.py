"""A Host is reached by its state, never by a deadline that load can reach, and admits light work without JAX."""

import asyncio
import os
import signal
import sys
import threading

import pytest

from feedbax.contracts.operation import DatasetPort, OperationDefinition
from feedbax.contracts.request_identity import encode_request
from feedbax.contracts.request_refs import InputMember, ProductRef
from feedbax.contracts.figure_document import FigureDocument
from feedbax.contracts.request_specs import AnalysisRequestSpec, RenderRequestSpec
from feedbax.host import AdmissionItem, Edge, HostClient, HostError, Job, Launch
from feedbax.host.protocol import LIGHT_ENTRYPOINT
from feedbax.light_jobs import LightDelivery
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.effective_success import store_request
from tests.support.hang import HANG_SECONDS


@pytest.mark.round(
    "A client keeps waiting, with no time limit of its own, while a Host holds its lock for 8 s before it listens."
)
def test_client_waits_for_a_host_that_holds_the_lock_before_it_listens(private_hosts):
    from feedbax.host.daemon import Host

    root = private_hosts / "host"

    async def main():
        host = Host(root)
        ping = asyncio.get_running_loop().run_in_executor(None, HostClient(root).ping)
        await asyncio.sleep(8)  # real clock: the client still waits after the Host has held its lock for 8 s
        waiting = not ping.done()
        task = asyncio.create_task(host.run())
        try:
            return waiting, await ping
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    waiting, reply = asyncio.run(main())
    assert waiting
    assert reply["pid"] == os.getpid()


def test_client_fails_as_soon_as_its_host_exits_at_start(private_hosts, monkeypatch):
    from feedbax.host import client as client_module

    root = private_hosts / "host"
    (root / "host.sqlite").mkdir(parents=True, mode=0o700)
    root.chmod(0o700)
    # The client's own wait for a starting Host outlasts the test's, so only the Host's exit can end it in time.
    monkeypatch.setattr(client_module, "HANG_SECONDS", 10 * HANG_SECONDS)
    errors = []

    def ping():
        try:
            HostClient(root).ping()
        except HostError as error:
            errors.append(error)

    pinging = threading.Thread(target=ping, daemon=True)
    pinging.start()
    pinging.join(HANG_SECONDS)
    assert not pinging.is_alive(), "The client waited past its Host's exit."
    (error,) = errors
    assert error.code == "unavailable" and "host.log" in str(error)


def test_host_admits_a_light_job_with_edges_without_importing_jax(tmp_path, monkeypatch, private_hosts):
    monkeypatch.setenv("PYTHONPROFILEIMPORTTIME", "1")
    blobs = ImmutableArtifactBlobProvider(tmp_path / "products")
    schema = "fixture.series.v1"
    definition = OperationDefinition(
        key="fixture.series", owner="fixture", owner_version="1", label="Series", description="Test input",
        param_schema=[], parameter_json_schema={},
        static_parameters=(), inputs={}, outputs={"series": DatasetPort(schema_ids=(schema,))},
    ).model_dump(mode="json")
    series = store_request(blobs, encode_request(AnalysisRequestSpec(analysis_type="fixture.series", params={}), {}))
    figures = FigureDocument.model_validate({
        "data": ["series"], "layout": [{"name": "main", "row": 1, "col": 1}],
        "layers": [{"name": "series", "role": "series", "panel": "main", "mark": "line",
                    "encode": {"x": "time", "y": "value"}}],
    })
    plot = store_request(blobs, encode_request(RenderRequestSpec(document=figures), {
        "series": (InputMember(key="data", reference=ProductRef(request_id=series.id, role="series")),),
    }))

    def job(attempt, request, parent=None):
        return Job(
            attempt, "project", request.id,
            Launch((sys.executable,), payload=LightDelivery(
                project_id="project", manifest_root=blobs.root, request=request,
            ).model_dump(mode="json"), warm_entrypoint=LIGHT_ENTRYPOINT, project_root=str(tmp_path)),
            "light", edges=(Edge(parent),) if parent else (),
            data={"manifest_root": str(blobs.root), "run_key": attempt},
        )

    client = HostClient(private_hosts / "host")
    pid = client.ping()["pid"]
    try:
        admitted = client.admit("light", [
            AdmissionItem("series", job("series", series), light_definition=definition),
            AdmissionItem("plot", job("plot", plot, "series")),
        ], record={})
        assert [item["disposition"] for item in admitted["items"]] == ["admitted", "admitted"]
        client.stop(["series", "plot"], "force")
    finally:
        os.kill(pid, signal.SIGTERM)
    imports = [line.rsplit("|", 1)[-1].strip() for line in (client.root / "host.log").read_text().splitlines()
               if line.startswith("import time:")]
    assert "feedbax.host.daemon" in imports
    assert "jax" not in imports

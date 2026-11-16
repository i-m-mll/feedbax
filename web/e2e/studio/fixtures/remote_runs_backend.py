"""Run the real Studio backend with a disposable venue and test attester."""

from contextlib import asynccontextmanager
from functools import partial
import os
from pathlib import Path
import sys
from time import monotonic, sleep

from feedbax.contracts.execution_target import KeepOpen, SessionTargetProfile
from feedbax.desktop.release import AppPaths
from feedbax.orchestration.session import SessionDriver
from feedbax.orchestration.session_jobs import SessionJobs
from feedbax.orchestration.venue_jobs import VenueJobs
from feedbax.web.app import create_app
from feedbax.web.services.remote_runs import StudioRemoteRuns, StudioVenue
from tests.support.session_venue import ENVIRONMENT, LocalVenueProvider, verify_authorization
from tests.support.store_transport import loopback_store
from tests.test_session_rehearsal import ISSUER_KEY, _authorization


def __getattr__(name: str):
    if name != "app":
        raise AttributeError(name)
    home = Path(os.environ["FEEDBAX_STUDIO_TEST_HOME"]).resolve()
    root = AppPaths.resolve(instance_root=home).state / "studio-e2e-remote"
    provider = LocalVenueProvider(root / "venues")
    driver = SessionDriver(root / "controller", provider, verify_authorization=partial(verify_authorization, key=ISSUER_KEY))
    profile = SessionTargetProfile(
        kind="session", id="rehearsal", label="Rehearsal Machine", driver=provider.target,
        resources={"compute": 1}, lifecycle=KeepOpen(), maximum_microusd=120_000,
    )
    connection = loopback_store(root / "stores", root / "ssh", download_gate=root / "download-gate").endpoint.connection

    def jobs(driver, session_id):
        state = driver.record(session_id)
        assert state.venue is not None
        venue = provider.directory(state.venue)
        return SessionJobs(driver, session_id, venue_jobs=VenueJobs(
            connection, python=sys.executable, cwd=Path.cwd(), host_root=venue / "host",
            projects_root=venue / "projects", stores_root=venue / "stores",
            worker_environment={"JAX_PLATFORMS": "cpu", "JAX_ENABLE_X64": "0"},
        ))

    def attest(quote, request):
        return _authorization(request, expires_at_ms=quote.expires_at_ms, ceiling_microusd=quote.ceiling_microusd)

    app = create_app(remote_runs=StudioRemoteRuns(
        (profile,), (StudioVenue(driver, ENVIRONMENT, jobs, 90),), attest=attest,
    ))
    application = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(app):
        try:
            async with application(app):
                yield
        finally:
            for state in driver.records():
                if state.phase != "open":
                    continue
                host = driver.attach(state.launch.request.id)
                running = host.session_status()["live_jobs"]
                if running:
                    host.stop(running, "force")
                    deadline = monotonic() + 8
                    while host.session_status()["live_jobs"]:
                        if monotonic() >= deadline:
                            raise TimeoutError("The isolated venue still has live workers.")
                        sleep(.05)
            for resource in provider.inventory():
                provider.remove(resource)

    app.router.lifespan_context = lifespan
    return app


if __name__ == "__main__":
    from scripts.studio_dev import main

    if "FEEDBAX_STUDIO_TEST_OWNER_PID" not in os.environ:
        raise SystemExit("The machine fixture requires an owning test runner.")
    raise SystemExit(main(backend_app="web.e2e.studio.fixtures.remote_runs_backend:app"))

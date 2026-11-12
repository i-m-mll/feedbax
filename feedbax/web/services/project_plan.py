"""Join the acknowledged compiler projection to observed Jobs attempts."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from feedbax.contracts.project_plan import RunKey
from feedbax.jobs import JobService, PlanJobAttempt, PlanJobAttempts
from feedbax.web.services.project_folder import ProjectFolderService, SettledProjectPlan


@dataclass(frozen=True)
class ProjectPlanRead:
    settled: SettledProjectPlan
    jobs: PlanJobAttempts
    attempts_by_run: Mapping[RunKey, tuple[PlanJobAttempt, ...]]


class ProjectPlanService:
    def __init__(self, folders: ProjectFolderService, jobs: JobService) -> None:
        self.folders = folders
        self.jobs = jobs

    def read(
        self, project_id: str, *, revision: str, source_runs: Iterable[RunKey] = ()
    ) -> ProjectPlanRead:
        """Read one saved revision and its request-scoped attempt evidence."""
        settled = self.folders.read_plan(project_id, revision=revision, source_runs=source_runs)
        requests = {run.request_id for run in settled.plan.runs if run.kind == "train"}
        jobs = self.jobs.read_plan_attempts(
            project=settled.project_id, store=settled.store_path, requests=requests
        )
        return ProjectPlanRead(
            settled,
            jobs,
            MappingProxyType({
                run.key: attempts
                for run in settled.plan.runs
                if (attempts := jobs.by_request.get(run.request_id)) is not None
            }),
        )

"""Execute one admitted Studio analysis operation in the owned worker process."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import traceback
from dataclasses import asdict
from pathlib import Path
from threading import Thread

from feedbax.web.services.analysis_service import write_job_json


def _watch_service(parent_fd: int) -> None:
    os.read(parent_fd, 1)
    os._exit(75)


async def _execute(path: Path) -> dict:
    from feedbax.contracts.figures import FigureSpec
    from feedbax.contracts.manifest import AnalysisRunSpec
    from feedbax.plugins.composition import compose_application
    from feedbax.web.api.analysis import _run_analysis_sync, _run_figure_sync

    payload = json.loads(path.read_bytes())
    bootstrap = await compose_application(modules=tuple(payload["bootstrap_modules"]))
    operation = payload["operation"]
    if operation["kind"] == "figure":
        result = _run_figure_sync(FigureSpec.model_validate(operation["spec"]), bootstrap.bundle)
    elif operation["kind"] == "analysis":
        spec = AnalysisRunSpec.model_validate(operation["spec"])
        computation = bootstrap.bundle.analysis_recipes.new_computation(spec.analysis_type)
        result = _run_analysis_sync(spec, bootstrap.bundle, computation=computation)
    else:
        raise ValueError("Unsupported Studio analysis operation")
    return asdict(result)


def main() -> None:
    job_dir, watch_fd, lease_fd = sys.argv[1:]
    os.fstat(int(lease_fd))
    Thread(target=_watch_service, args=(int(watch_fd),), daemon=True).start()
    root = Path(job_dir)
    try:
        result = asyncio.run(_execute(root / "operation.json"))
    except Exception:
        result = {"error": traceback.format_exc()}
    write_job_json(root / "result.json", result)


if __name__ == "__main__":
    main()

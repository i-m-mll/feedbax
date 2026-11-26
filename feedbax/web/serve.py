"""Studio's backend entry, run with the interpreter of the environment it serves.

It restarts when a source file of an editable project package changes, so plans, catalogs and Problems always come
from the code a job would run. The listening socket belongs to the launcher, so a request made during a restart waits.
"""

from __future__ import annotations

import argparse
from importlib.util import find_spec
import os
from pathlib import Path

from fastapi import FastAPI
import uvicorn

from feedbax.host.platform import end_with
from feedbax.plugins.bootstrap import installed_packages
from feedbax.project_environment import editable_source

OWNER_VARIABLE = "FEEDBAX_STUDIO_OWNER"
OPEN_VARIABLE = "FEEDBAX_STUDIO_OPEN"
STUDIO_APP = "feedbax.web.serve:studio_app"


def _end_with_owner() -> None:
    if (owner := os.environ.get(OWNER_VARIABLE)) is not None:
        end_with(int(owner))


def studio_app() -> FastAPI:
    from feedbax.web.app import create_app

    _end_with_owner()
    folder = os.environ.get(OPEN_VARIABLE)
    return create_app(open_folder=Path(folder) if folder else None)


def project_package_sources() -> list[Path]:
    """The source directories of the editable packages that supply declarations, Feedbax's own excluded."""
    directories = []
    for source in installed_packages():
        if source.distribution == "feedbax" or editable_source(source.distribution) is None:
            continue
        spec = find_spec(source.package.partition(".")[0])
        if spec is not None and spec.submodule_search_locations:
            directories.extend(Path(location) for location in spec.submodule_search_locations)
    return directories


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fd", type=int, required=True, help="serve on this inherited listening socket")
    parser.add_argument("--owner", type=int, required=True, help="exit when this process exits")
    parser.add_argument("--open", type=Path, help="open this project folder in place at every start")
    parser.add_argument("--app", default=STUDIO_APP, help="serve this application instead of Studio's")
    args = parser.parse_args(argv)
    os.environ[OWNER_VARIABLE] = str(args.owner)
    if args.open is not None:
        os.environ[OPEN_VARIABLE] = str(args.open)
    _end_with_owner()
    watched = project_package_sources()
    uvicorn.run(
        args.app, factory=args.app == STUDIO_APP, fd=args.fd,
        reload=bool(watched), reload_dirs=[str(directory) for directory in watched],
    )


if __name__ == "__main__":
    main()

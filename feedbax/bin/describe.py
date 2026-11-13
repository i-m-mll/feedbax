"""Render the shared authoring catalog and its individual entries as JSON."""

import argparse
import asyncio
import json

from pydantic import BaseModel

from feedbax.bin.project_commands import CommandOutput
from feedbax.contracts.studio_authoring import StudioAuthoringCatalog


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "selector", nargs="?",
        help="Family, step kind or registry. Qualify with family:, step: or registry: as needed.",
    )
    parser.add_argument(
        "--plugin", action="append", help="Load a plugin module's declarations. May be repeated.",
    )


def _select(catalog: StudioAuthoringCatalog, selector: str) -> tuple[str, str]:
    aliases = {
        entry.schema_id: name
        for name, entry in catalog.document_families.items()
        if entry.schema_id is not None
    }
    fields = {"step": "step_payloads", "family": "document_families", "registry": "registries"}
    kind, separator, name = selector.partition(":")
    for candidate_kind, field in fields.items():
        if separator and candidate_kind != kind:
            continue
        lookup = name if separator else selector
        entries = getattr(catalog, field)
        candidate = (
            aliases.get(lookup, lookup)
            if candidate_kind == "family" and lookup not in entries else lookup
        )
        if candidate in entries:
            return field, candidate
    available = [
        f"{kind}:{name}"
        for kind, field in fields.items()
        for name in getattr(catalog, field)
    ]
    available.extend(f"family:{name}" for name in aliases)
    raise ValueError(
        f"unknown authoring selector {selector!r}. Available selectors: {', '.join(available)}"
    )


def run(args: argparse.Namespace, output: CommandOutput) -> int:
    from feedbax.plugins.bootstrap import BootstrapError
    from feedbax.plugins.composition import compose_application
    from feedbax.studio.authoring import authoring_surface

    try:
        state = asyncio.run(compose_application(modules=tuple(args.plugin or ())))
    except BootstrapError as error:
        output.failure(error, path="--plugin")
        return 2
    catalog = authoring_surface(state, catalog=True)
    result = catalog
    if args.selector is not None:
        try:
            field, name = _select(catalog, args.selector)
        except ValueError as error:
            output.error("unknown_selector", str(error), address=args.selector, path=args.selector)
            return 2
        entry = getattr(catalog, field)[name]
        if output.json:
            result = catalog.model_copy(update={
                "step_payloads": {}, "document_families": {}, "registries": {}, field: {name: entry},
            })
        else:
            result = entry
    if output.json:
        output.response(result, exclude_unset=False)
    else:
        payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
        output.write(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0

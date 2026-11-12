"""Render the shared authoring catalog and its individual entries as JSON."""

import json
import sys

from pydantic import BaseModel

from feedbax.contracts.studio_authoring import StudioAuthoringCatalog
from feedbax.plugins.bootstrap import BootstrapState
from feedbax.studio.authoring import authoring_surface


def _select(catalog: StudioAuthoringCatalog, selector: str) -> object:
    families = catalog.document_families | {
        entry.schema_id: entry
        for entry in catalog.document_families.values()
        if entry.schema_id is not None
    }
    choices = {
        "step": catalog.step_payloads,
        "family": families,
        "registry": catalog.registries,
    }
    kind, separator, name = selector.partition(":")
    if separator:
        if kind in choices and name in choices[kind]:
            return choices[kind][name]
    else:
        for entries in choices.values():
            if selector in entries:
                return entries[selector]
    available = ", ".join(f"{kind}:{name}" for kind, entries in choices.items() for name in entries)
    raise ValueError(f"unknown authoring selector {selector!r}. Available selectors: {available}")


def describe(state: BootstrapState, selector: str | None) -> int:
    catalog = authoring_surface(state, catalog=True)
    try:
        result = _select(catalog, selector) if selector is not None else catalog
    except ValueError as error:
        print(f"feedbax describe: {error}", file=sys.stderr)
        return 2
    payload = result.model_dump(mode="json") if isinstance(result, BaseModel) else result
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0

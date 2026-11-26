"""Shared expression site and function tables for one Studio field response."""

from collections.abc import Sequence
from typing import TypedDict

from feedbax.contracts.expression_site import FieldKind, SiteProjection, initial_draw_per
from feedbax.contracts.studio_field import StudioFieldFunction, StudioFieldName, StudioFieldSite, StudioFieldView


class _Tables(TypedDict):
    sites: dict[str, StudioFieldSite]
    function_lists: dict[str, tuple[StudioFieldFunction, ...]]


class FieldSiteIndex:
    def __init__(self) -> None:
        self._sites: dict[str, StudioFieldSite] = {}
        self._function_lists: dict[str, tuple[StudioFieldFunction, ...]] = {}
        self._site_ids: dict[tuple[object, ...], str] = {}
        self._function_list_ids: dict[tuple[str, ...], str] = {}

    def reference(self, projection: SiteProjection) -> str:
        function_ids = tuple(info.id for info in projection.functions)
        functions = self._function_list_ids.get(function_ids)
        if functions is None:
            functions = str(len(self._function_lists))
            self._function_list_ids[function_ids] = functions
            self._function_lists[functions] = tuple(
                StudioFieldFunction.model_validate(info, from_attributes=True) for info in projection.functions
            )
        site = projection.site
        names = tuple(
            (name.id, name.name, name.type.base, name.type.shape, name.type.space, name.list_count, name.set_entry)
            for name in site.names
        )
        signature = (
            site.expected, site.finest, names, site.structural, site.list_count, site.repeat_count,
            projection.kinds, functions, projection.parts, tuple(sorted(projection.name_paths.items())),
        )
        key = self._site_ids.get(signature)
        if key is None:
            key = str(len(self._sites))
            self._site_ids[signature] = key
            self._sites[key] = StudioFieldSite(
                expected=site.expected, finest=site.finest,
                names=tuple(StudioFieldName.model_validate(name, from_attributes=True) for name in site.names),
                structural=site.structural, list_count=site.list_count, repeat_count=site.repeat_count,
                kinds=projection.kinds, functions=functions, parts=projection.parts, name_paths=projection.name_paths,
                draw_per=initial_draw_per(site.finest) if FieldKind.DRAW in projection.kinds else None,
            )
        return key

    def tables(self, fields: Sequence[StudioFieldView]) -> _Tables:
        sites = {field.site: self._sites[field.site] for field in fields}
        return {
            "sites": sites,
            "function_lists": {site.functions: self._function_lists[site.functions] for site in sites.values()},
        }

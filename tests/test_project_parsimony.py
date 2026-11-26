"""A stored key must affect what the project reads, and each write must be stable."""

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from functools import reduce
import json
from operator import getitem
from pathlib import Path

from pydantic import BaseModel, ValidationError
import pytest

from feedbax.contracts.canonical_json import CanonicalJsonError, canonical_json_v2_bytes
from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project, RootDocument
from feedbax.contracts.project_io import (
    STUDIO_PRESENTATION_FILENAME, ProjectConfiguration, ProjectReadError, dump_text_project, parse_text_project_files,
    TextProjectManifest,
)
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.project_storage import stored_defaults_manifest, stored_document_models, write_project_file
from feedbax.contracts.project_sweep import AxisValue, NamedAxis, Row, Sweep, SweepException
from feedbax.contracts.studio_project import PresentationDocument
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID
from feedbax.planner.project_plan import project_variants

_SHA = "d" * 64
_OBJECT = {"object_id": f"content-object://sha256/{_SHA}", "sha256": _SHA, "size_bytes": 42}
_TASK = {
    "id": "task", "variables": {}, "signals": {}, "fixed_steps": None,
    "epochs": {"trial": {"label": "Trial", "end": {"kind": "length", "length": {"kind": "literal", "value": 1}}}},
}
_DOCUMENT_FIXTURES = {
    STORED_MODEL_DOCUMENT_SCHEMA_ID: {"step_seconds": 0.01, "ports": {}, "interior": {}},
    STORED_TASK_DOCUMENT_SCHEMA_ID: _TASK,
    "feedbax.spec.python_task": {"type": "feedbax.example_task", "params": {}},
    STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID: {"id": "score", "terms": {
        "effort": {"enabled": True, "weight": {"kind": "literal", "value": 1.0},
         "measure": {"kind": "literal", "value": 2.0}, "reduction": "mean"},
    }},
    "feedbax.spec.training_document": {
        "method": "feedbax.example_training", "settings": {"schedule": {"value": False, "scale": 1.0}},
        "training_config": {"batch_size": 128, "snapshot_interval": 100, "replicates": 1},
    },
    "feedbax.spec.loaded_model": {
        "schema_version": "feedbax.spec.loaded_model.v1", "model": {"document": "model"}, "archive": None,
        "source": {"kind": "imported", "model": _OBJECT,
                   "weights": {"tree_id": f"content-tree://sha256/{_SHA}", "descriptor": _OBJECT},
                   "encoding": {"descriptor_sha256": _SHA}},
    },
    "feedbax.set_document.v2": {"entries": {"nominal": {"label": "Nominal", "differences": {}}}},
    "feedbax.analysis_document.v1": {
        "inputs": {"states": {"kind": "dataset", "cardinality": "many"}},
        "operations": {"align": {
            "label": "Align", "operation": "feedbax.aligned_velocity", "params": {}, "inputs": {"states": {"input": "states"}},
        }},
    },
    "feedbax.figure_document.v1": {
        "data": ["summary"], "theme": "plotly_white", "layout": [{"name": "velocity", "row": 1, "col": 1}],
        "layers": [{"name": "bounds", "mark": "band", "role": "summary", "panel": "velocity",
                    "encode": {"x": "time", "y": "mean", "lower": "lower", "upper": "upper"}}],
    },
    "feedbax.report_document.v1": {
        "title": "Report", "blocks": [{"markdown": "## Summary"}, {"figure": "velocity", "caption": None}],
        "completeness": "complete",
    },
    "subset": {"rule": "", "clauses": [{"noise": [{"id": "nominal", "label": "Nominal"}]}]},
}

type _Path = tuple[str | int, ...]
type _Observe = Callable[[Project], object]


def _refinement_fields(value: object, path: _Path = ()) -> Iterator[tuple[_Path, tuple[str, ...]]]:
    if isinstance(value, BaseModel):
        if isinstance(value, (Sweep, NamedAxis, AxisValue, Row, SweepException)):
            yield path, tuple(sorted(value.model_fields_set))
        for field in type(value).model_fields:
            yield from _refinement_fields(getattr(value, field), (*path, field))
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from _refinement_fields(child, (*path, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _refinement_fields(child, (*path, index))


def _meaning(model: BaseModel) -> tuple[bytes, tuple]:
    """Read values with defaults filled, retaining which refinement fields the author set."""
    value = model.model_dump(mode="json", by_alias=True)
    if isinstance(model, RootDocument) and (payload_model := stored_document_models()[model.schema_id]) is not None:
        value["payload"] = payload_model.model_validate(model.payload).model_dump(mode="json", by_alias=True)
    return canonical_json_v2_bytes(value), tuple(_refinement_fields(model))


def _read(files: Mapping[str, bytes], root: Path, observe: _Observe | None):
    text = parse_text_project_files(root, files)
    presentation = PresentationDocument.model_validate(json.loads(files[STUDIO_PRESENTATION_FILENAME]))
    project = text.project
    meaning = (
        _meaning(text.configuration), _meaning(text.manifest), _meaning(presentation),
        {key: _meaning(value) for key, value in project.documents.items()},
        {key: _meaning(value) for key, value in project.steps.items()},
        _meaning(project.sweep) if project.sweep is not None else None,
        observe(project) if observe is not None else None,
    )
    return text, presentation, meaning


def _positions_of_keys(value: object, prefix: _Path = ()) -> Iterator[_Path]:
    if isinstance(value, dict):
        for key, child in value.items():
            path = (*prefix, key)
            yield path
            yield from _positions_of_keys(child, path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _positions_of_keys(child, (*prefix, index))


def _verify_parsimony(root: Path, project: Project, *, observe: _Observe | None = None) -> dict[str, bytes]:
    configuration = ProjectConfiguration(project="fixture", store="results", settings={})
    presentation = PresentationDocument.model_validate({
        "documents": {"model": {"positions": {"point": {"x": -0.0, "y": 1.0}}, "flipped": {}}},
    })
    files = dump_text_project(project, configuration) | {STUDIO_PRESENTATION_FILENAME: write_project_file(presentation)}
    reopened, shown, expected = _read(files, root, observe)
    assert {key: _meaning(value) for key, value in reopened.project.documents.items()} == {
        key: _meaning(value) for key, value in project.documents.items()
    }
    if observe is not None:
        assert observe(reopened.project) == observe(project)
    rewritten = dump_text_project(reopened.project, reopened.configuration)
    assert rewritten | {STUDIO_PRESENTATION_FILENAME: write_project_file(shown)} == files

    for relative, data in files.items():
        original = json.loads(data)
        for path in _positions_of_keys(original):
            edited = deepcopy(original)
            del reduce(getitem, path[:-1], edited)[path[-1]]
            candidate = files | {relative: json.dumps(edited).encode()}
            try:
                _, _, changed = _read(candidate, root, observe)
            except (ProjectReadError, ValidationError):
                continue
            assert changed != expected, f"{relative} repeats a default at {path}"
    return files


@pytest.mark.parametrize("schema", sorted(stored_document_models()))
def test_each_registered_document_has_a_parsimonious_file(tmp_path, schema):
    assert schema in _DOCUMENT_FIXTURES, f"Register a document harness fixture for {schema}."
    document = RootDocument(schema=schema, label="Fixture", payload=_DOCUMENT_FIXTURES[schema])
    files = _verify_parsimony(tmp_path, Project(documents={"fixture": document}))
    if schema == STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID:
        # The existing expression tree's scalar type can change whether the checker accepts it.
        term = json.loads(files["documents/fixture.json"])["payload"]["terms"]["effort"]
        assert type(term["measure"]["value"]) is float


def test_stored_defaults_have_a_pinned_schema_contract():
    path = Path(__file__).resolve().parents[1] / "feedbax/contracts/stored_defaults_contract.json"
    assert canonical_json_v2_bytes(stored_defaults_manifest()) == canonical_json_v2_bytes(json.loads(path.read_bytes()))


def test_refinement_keeps_the_default_an_author_explicitly_set(tmp_path):
    project = Project.model_validate({
        "steps": {
            "base": {"kind": "train", "label": "Base", "base": {"inline": {}}, "sweep": {
                "axes": {"treatment": {"values": {"one": {"label": "One"}}, "optional": True}},
            }},
            "child": {"kind": "train", "label": "Child", "base": {"step": "base"}, "sweep": {
                "axes": {"treatment": {"values": {"one": {"label": None}}, "optional": False}},
            }},
        },
        "sweep": {"rows": {"baseline": {"patches": {}}, "check": {"label": "Check"}}},
    })
    files = _verify_parsimony(tmp_path, project)
    treatment = json.loads(files["steps/child.json"])["sweep"]["axes"]["treatment"]
    assert treatment["optional"] is False
    assert treatment["values"]["one"]["label"] is None


def test_a_default_retains_its_presence_when_other_files_observe_it(tmp_path):
    project = Project.model_validate({
        "documents": {
            "base": {"schema": STORED_TASK_DOCUMENT_SCHEMA_ID, "label": "Base", "payload": _TASK},
            "derived": {
                "schema": STORED_TASK_DOCUMENT_SCHEMA_ID, "label": "Derived", "base": {"document": "base"},
                "differences": {"/fixed_steps": {"op": "remove"}},
            },
        },
        "sweep": {"rows": {"longer": {"patches": {"documents.base.payload.fixed_steps": {"op": "replace", "value": 5}}}}},
    })

    def observe(project: Project):
        derived = compose_document(project.documents.get, "derived")
        variants = project_variants(project)
        changed = variants.get("longer")
        limit = changed.documents["base"].payload["fixed_steps"] if changed is not None else None
        return not isinstance(derived, ProjectDiagnostic), limit

    assert observe(project) == (True, 5)
    files = _verify_parsimony(tmp_path, project, observe=observe)
    assert json.loads(files["documents/base.json"])["payload"]["fixed_steps"] is None


def test_opaque_patch_values_keep_types_the_legacy_expression_checker_reads(tmp_path):
    schema = STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
    base = _DOCUMENT_FIXTURES[schema]
    path = "documents.base.payload.terms.effort.measure.value"
    project = Project.model_validate({
        "documents": {
            "base": {"schema": schema, "label": "Base", "payload": base},
            "derived": {
                "schema": schema, "label": "Derived", "base": {"document": "base"},
                "differences": {"/terms": {"value": base["terms"]}},
            },
        },
        "sweep": {"rows": {"larger": {"patches": {path: {"op": "replace", "value": 3.0}}}}},
    })
    files = _verify_parsimony(tmp_path, project)
    terms = json.loads(files["documents/derived.json"])["differences"]["/terms"]["value"]
    assert type(terms["effort"]["measure"]["value"]) is float
    assert type(json.loads(files["variants/larger.json"])["patches"][path]["value"]) is float


def test_project_numbers_round_trip_without_rounding_or_accepting_unsafe_integers():
    fields = {"near": 0.00030000000000000003, "large": 1e20, "whole_real": 1.0, "zero": -0.0, "flag": False}
    manifest = TextProjectManifest(schema_version="feedbax.project.v1", store="results", settings=fields)
    encoded = write_project_file(manifest)
    reopened = TextProjectManifest.model_validate(json.loads(encoded))
    assert reopened.settings == fields
    assert type(reopened.settings["large"]) is float
    assert type(reopened.settings["whole_real"]) is int
    assert type(reopened.settings["zero"]) is int
    assert reopened.settings["flag"] is False
    assert write_project_file(reopened) == encoded

    invalid = manifest.model_copy(update={"settings": {"unsafe": 2**53}})
    with pytest.raises(CanonicalJsonError, match="integer must be between"):
        write_project_file(invalid)


def test_project_layout_counts_code_points_and_keeps_long_strings_intact():
    settings = {
        "matrix": [list(range(6)) for _ in range(6)],
        "filled": list(range(40)),
        "inline": ["λ" * 74],
        "long": "λ" * 110,
    }
    manifest = TextProjectManifest(schema_version="feedbax.project.v1", store="results", settings=settings)
    encoded = write_project_file(manifest).decode()
    assert json.loads(encoded)["settings"] == settings
    assert list(json.loads(encoded)["settings"]) == list(settings)

    lines = encoded.splitlines()
    matrix_start = lines.index('    "matrix": [')
    assert lines[matrix_start + 7] == "    ],"
    filled_start = lines.index('    "filled": [')
    inline_start = next(index for index, line in enumerate(lines) if '"inline"' in line)
    assert inline_start - filled_start == 4
    assert len(lines[inline_start].encode()) > 100 >= len(lines[inline_start])
    assert [line.split('"')[1] for line in lines if len(line) > 100] == ["long"]

"""A stored key must affect what the project reads, and each write must be stable."""

from collections.abc import Callable, Iterator, Mapping
from copy import deepcopy
from functools import partial, reduce
import json
from operator import getitem
from pathlib import Path
from typing import get_args

from pydantic import BaseModel, RootModel, ValidationError
import pytest

from feedbax.contracts.analysis_document import ANALYSIS_SCHEMA
from feedbax.contracts.canonical_json import CanonicalJsonError, canonical_json_v2_bytes
from feedbax.contracts.loaded_model import LOADED_MODEL_SCHEMA_ID
from feedbax.contracts.figure_document import STORED_FIGURE_DOCUMENT_SCHEMA_ID, StoredFigureDocument
from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import (
    BasedDocument, InlineBase, Project, RootDocument, Step, StepKind, apply_differences, diff,
)
from feedbax.contracts.project_io import (
    STUDIO_PRESENTATION_FILENAME, ProjectConfiguration, dump_text_project, parse_text_project_files,
    TextProjectManifest,
)
from feedbax.contracts.project_resolution import ProjectDiagnostic, compose_document
from feedbax.contracts.project_storage import (
    stored_defaults_manifest, stored_document_models, stored_keyed_collection, stored_project_value, write_project_file,
)
from feedbax.contracts.project_sweep import AxisValue, NamedAxis, Row, Sweep, SweepException
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.contracts.stored_regions import StoredRegion, StoredRegionRegistries, step_inline_model, typed_region
from feedbax.contracts.studio_project import PresentationDocument
from feedbax.contracts.stored_model import STORED_MODEL_DOCUMENT_SCHEMA_ID
from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID
from feedbax.planner.project_plan import project_variants
from tests.support.python_task import REACHING_TASK, reaching_python_task  # noqa: F401

_SHA = "d" * 64
_OBJECT = {"sha256": _SHA, "size_bytes": 42}
_TASK = {
    "id": "task", "variables": {}, "signals": {}, "fixed_steps": None,
    "epochs": {"trial": {"label": "Trial", "end": {"kind": "length", "length": {"kind": "literal", "value": 1}}}},
}
_FIGURE = {
    "data": ["summary"], "theme": "plotly_white",
    "layout": {"velocity": {"row": 1, "col": 1}, "markers": {"row": 1, "col": 2}},
    "layers": {
        "mean": {"mark": "line", "role": "summary", "panel": "velocity", "encode": {"x": "time", "y": "mean"}},
        "bounds": {
            "mark": "band", "role": "summary", "panel": "velocity", "style": {"band_alpha": 0.2},
            "encode": {"x": "time", "y": "mean", "lower": "lower", "upper": "upper"},
        },
        "zero": {"mark": "rule", "role": "summary", "panel": "markers", "encode": {"axis": "x", "at": 0.0}},
    },
}
_TIME = {"kind": "literal", "value": 0.2, "unit": "s"}
_MODEL = {
    "step_seconds": 0.01, "ports": {}, "interior": {"components": {
        "cell": {"kind": "leaf", "label": "Cell", "type": "feedbax.VanillaRNN", "fields": {"hidden_size": 4, "tau": _TIME}},
    }},
}
_METHOD = "feedbax.supervised_gradient"
_TRAIN_SETTINGS = {
    "grad_clip": 1.0,
    "optimizer": {"type": "adam", "learning_rate": {"type": "constant_schedule", "value": 0.01}, "b1": 0.9},
}
_ANALYSIS = {
    "inputs": {"states": {"kind": "dataset", "cardinality": "many"}},
    "operations": {"summary": {
        "label": "Velocity", "operation": "feedbax.summarize_velocity",
        "params": {"ddof": 0, "bounds_scale": 1.0, "variable": "parallel_velocity"},
        "inputs": {"states": {"input": "states"}},
    }},
    "views": {"profile": {"label": "Profile", "figure": _FIGURE, "inputs": {"summary": {"input": "states"}}}},
}
_REPORT = {
    "title": "Report", "blocks": [{"markdown": "## Summary"}, {"figure": "velocity", "caption": None}],
    "completeness": "complete",
}
_DOCUMENT_FIXTURES = {
    STORED_MODEL_DOCUMENT_SCHEMA_ID: _MODEL,
    STORED_TASK_DOCUMENT_SCHEMA_ID: _TASK,
    "feedbax.spec.python_task": {"type": REACHING_TASK, "params": {"n_steps": 50, "x": 0.15, "y": 0.0}},
    STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID: {"id": "score", "terms": {
        "effort": {"enabled": True, "weight": {"kind": "literal", "value": 1.0},
         "measure": {"kind": "literal", "value": 2.0}, "reduction": "mean"},
    }},
    "feedbax.spec.training_document": {
        "method": _METHOD, "settings": _TRAIN_SETTINGS,
        "training_config": {"batch_size": 128, "snapshot_interval": 100, "replicates": 1},
    },
    LOADED_MODEL_SCHEMA_ID: {
        "model": {"document": "model"}, "model_revision": _OBJECT, "archive": None,
        "weights": {"sha256": "e" * 64, "size_bytes": 54}, "encoding_sha256": "f" * 64,
    },
    "feedbax.set_document.v2": {"entries": {"nominal": {
        "label": "Nominal",
        "differences": {
            "/interior/components/cell/fields/tau": {"value": _TIME},
            "/interior/components/cell/fields/hidden_size": {"value": 4},
        },
    }}},
    ANALYSIS_SCHEMA: _ANALYSIS,
    STORED_FIGURE_DOCUMENT_SCHEMA_ID: _FIGURE,
    "feedbax.report_document.v1": _REPORT,
    "subset": {"rule": "", "clauses": [{"noise": [{"id": "nominal", "label": "Nominal"}]}]},
}

_TRIALS = {"model": _MODEL, "task": _TASK, "objective": _DOCUMENT_FIXTURES[STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID]}
_STEP_FIXTURES = {
    "train": {"method": _METHOD, "training_config": {"n_batches": 3, "replicates": 1}, "settings": _TRAIN_SETTINGS,
              "trial_documents": _TRIALS | {"input_bindings": {}}},
    "synthesize": _TRIALS | {"method": "feedbax.constant_parameters", "settings": {"value": 0.0}},
    "evaluate": {"evaluation_type": ROLLOUT_TYPE, "params": {
        "seed": 7, "trial_policy": "paired", "variation": [], "retention": "raw_states",
        "trial_set": {"kind": "every_condition"},
    }},
    "analyze": _ANALYSIS,
    "report": _REPORT,
}
_PRESENTATION = {
    "documents": {"model": {
        "positions": {"component.point": {"x": -0.0, "y": 1.0}},
        "flipped": {"component.point": True},
        "collapsed": {"controller/graph": True},
        "waypoints": {"wire.route": [{"x": 1.0, "y": 2.0}]},
        "camera": {"x": 0, "y": 0, "zoom": 0.5},
    }},
    "steps": {"train": {"sweep_order": ["length"], "model_positions": {"model": {"component.point": {"x": 2, "y": 0}}}}},
    "visible_groups": ["trial"],
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


def _effective_payload(value, encoded, registries):
    """Interpret stored values with registered types and defaults before comparing reads."""
    if isinstance(value, RootModel):
        return _effective_payload(value.root, encoded, registries)
    if isinstance(value, RootDocument):
        model = stored_document_models()[value.schema_id]
        assert model is not None, f"{value.schema_id} has no stored payload model"
        payload = model.model_validate(value.payload)
        return encoded | {"payload": _effective_payload(payload, payload.model_dump(mode="json", by_alias=True), registries)}
    if isinstance(value, BaseModel):
        def read_field(name, field):
            key = field.serialization_alias or field.alias or name
            child = getattr(value, name)
            if isinstance(value, Step) and isinstance(child, InlineBase):
                region = StoredRegion(step_inline_model(value.kind, child.inline))
                return {"inline": read_region(region, child.inline, inline=True)}
            if isinstance(encoded[key], dict) and (region := registries.region(value, name)) is not None:
                return read_region(region, encoded[key])
            return _effective_payload(child, encoded[key], registries)

        def read_region(region, raw, *, inline=False):
            if region.model is None:
                return raw
            context = {key: item for key, item in raw.items() if key in region.contextual}
            typed, values, _ = typed_region(region, {key: item for key, item in raw.items() if key not in context}, inline=inline)
            return _effective_payload(typed, values, registries) | context

        for name, field in type(value).model_fields.items():
            key = field.serialization_alias or field.alias or name
            if key in encoded:
                encoded[key] = read_field(name, field)
        return encoded
    if isinstance(value, dict) and isinstance(encoded, dict):
        return {key: _effective_payload(value.get(key), child, registries) for key, child in encoded.items()}
    if isinstance(value, (tuple, list)) and isinstance(encoded, list):
        return [_effective_payload(item, child, registries) for item, child in zip(value, encoded, strict=True)]
    return encoded


def _meaning(model: BaseModel) -> tuple[bytes, tuple]:
    """Read values with defaults filled, retaining which refinement fields the author set."""
    encoded = model.model_dump(mode="json", by_alias=True)
    value = _effective_payload(model, encoded, StoredRegionRegistries())
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
    presentation = PresentationDocument.model_validate(_PRESENTATION)
    files = dump_text_project(project, configuration) | {STUDIO_PRESENTATION_FILENAME: write_project_file(presentation)}
    reopened, shown, expected = _read(files, root, observe)
    assert {key: _meaning(value) for key, value in reopened.project.documents.items()} == {
        key: _meaning(value) for key, value in project.documents.items()
    }
    assert {key: _meaning(value) for key, value in reopened.project.steps.items()} == {
        key: _meaning(value) for key, value in project.steps.items()
    }
    if observe is not None:
        assert observe(reopened.project) == observe(project)
    rewritten = dump_text_project(reopened.project, reopened.configuration)
    assert rewritten | {STUDIO_PRESENTATION_FILENAME: write_project_file(shown)} == files

    _verify_essential_keys(files, root, observe, expected)
    return files


def _verify_essential_keys(files, root, observe, expected):
    for relative, data in files.items():
        original = json.loads(data)
        for path in _positions_of_keys(original):
            edited = deepcopy(original)
            del reduce(getitem, path[:-1], edited)[path[-1]]
            candidate = files | {relative: json.dumps(edited).encode()}
            try:
                _, _, changed = _read(candidate, root, observe)
            except ValueError:
                continue
            assert changed != expected, f"{relative} repeats a default at {path}"


@pytest.mark.usefixtures("reaching_python_task")
@pytest.mark.parametrize("schema", sorted(stored_document_models()))
def test_each_registered_document_has_a_parsimonious_file(tmp_path, schema):
    assert stored_document_models()[schema] is not None, f"{schema} has no stored payload model"
    assert schema in _DOCUMENT_FIXTURES, f"Register a document harness fixture for {schema}."
    document = RootDocument(schema=schema, label="Fixture", payload=_DOCUMENT_FIXTURES[schema])
    files = _verify_parsimony(tmp_path, Project(documents={"fixture": document}))
    if schema == STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID:
        # The existing expression tree's scalar type can change whether the checker accepts it.
        term = json.loads(files["documents/fixture.json"])["payload"]["terms"]["effort"]
        assert type(term["measure"]["value"]) is float


@pytest.mark.parametrize("kind", get_args(StepKind.__value__))
def test_each_inline_step_has_a_parsimonious_file(tmp_path, kind):
    assert kind in _STEP_FIXTURES, f"Register an inline step harness fixture for {kind}."
    step = Step.model_validate({"kind": kind, "label": kind.title(), "base": {"inline": _STEP_FIXTURES[kind]}})
    _verify_parsimony(tmp_path, Project(steps={kind: step}))


def test_set_fields_deltas_and_variant_refinements_are_parsimonious(tmp_path):
    project = Project.model_validate({
        "documents": {"perturbations": {
            "schema": "feedbax.set_document.v2",
            "label": "Perturbations",
            "payload": _DOCUMENT_FIXTURES["feedbax.set_document.v2"],
        }},
        "steps": {"train": {
            "kind": "train",
            "label": "Train",
            "base": {"inline": _STEP_FIXTURES["train"] | {"seed": 0}},
            "set_fields": {"perturbation": {
                "axis": {"set": "perturbations", "optional": False},
                "sampling": {"kind": "weighted", "weights": {"nominal": 2.0}, "none": 1.0},
                "condition": {"kind": "literal", "value": True},
            }},
            "deltas": [{
                "layer_id": "seed",
                "patches": [{"path": "seed", "value": 7}],
                "acknowledges_ancestor_paths": [],
                "schema_id": None,
                "schema_version": None,
            }],
            "sweep": {"axes": {"length": {"values": {"long": {
                "patches": {"training_config.n_batches": {"value": 4}},
            }}}}},
        }},
        "sweep": {"rows": {"longer": {"patches": {"steps.train.base.inline.training_config.n_batches": {"value": 5}}}}},
    })
    files = _verify_parsimony(tmp_path, project)
    train = json.loads(files["steps/train.json"])
    assert train["base"]["inline"]["seed"] == 0
    assert json.loads(files["variants/longer.json"])["patches"]


def test_a_based_objective_retains_the_expression_and_default_it_writes(tmp_path):
    schema = STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
    term = deepcopy(_DOCUMENT_FIXTURES[schema]["terms"]["effort"])
    project = Project.model_validate({"documents": {
        "objective": {"schema": schema, "label": "Objective", "payload": _DOCUMENT_FIXTURES[schema]},
        "alternative": {"schema": schema, "label": "Alternative", "base": {"document": "objective"},
                        "differences": {"/terms/alternate": {"value": term}}},
    }})

    def observe(project: Project):
        document = compose_document(project.documents.get, "alternative")
        return _meaning(document) if isinstance(document, RootDocument) else document

    files = _verify_parsimony(tmp_path, project, observe=observe)
    derived = json.loads(files["documents/alternative.json"])
    assert derived["differences"]["/terms/alternate"]["value"] == term


@pytest.mark.usefixtures("reaching_python_task")
@pytest.mark.parametrize(("family", "name", "path", "value"), [
    ("documents", STORED_MODEL_DOCUMENT_SCHEMA_ID, ("payload", "interior", "components", "cell", "fields", "hidden_size"), 4),
    ("documents", "feedbax.spec.python_task", ("payload", "params", "x"), 0.15),
    ("documents", "feedbax.spec.training_document", ("payload", "settings", "grad_clip"), 1.0),
    ("documents", ANALYSIS_SCHEMA, ("payload", "operations", "summary", "params", "bounds_scale"), 1.0),
    ("documents", STORED_FIGURE_DOCUMENT_SCHEMA_ID, ("payload", "theme"), "plotly_white"),
    ("steps", "synthesize", ("base", "inline", "settings"), {"value": 0.0}),
    ("steps", "evaluate", ("base", "inline", "params", "trial_policy"), "paired"),
])
def test_the_harness_refuses_a_restated_registry_default(tmp_path, family, name, path, value):
    if family == "documents":
        document = RootDocument(schema=name, label="Fixture", payload=_DOCUMENT_FIXTURES[name])
        project = Project(documents={"fixture": document})
    else:
        step = Step.model_validate({"kind": name, "label": "Fixture", "base": {"inline": _STEP_FIXTURES[name]}})
        project = Project(steps={"fixture": step})
    files = _verify_parsimony(tmp_path, project)
    relative = f"{family}/fixture.json"
    authored = json.loads(files[relative])
    reduce(getitem, path[:-1], authored)[path[-1]] = value
    redundant = files | {relative: json.dumps(authored).encode()}
    _, _, expected = _read(redundant, tmp_path, None)
    with pytest.raises(AssertionError, match="repeats a default"):
        _verify_essential_keys(redundant, tmp_path, None, expected)


def test_stored_defaults_have_a_pinned_schema_contract():
    path = Path(__file__).resolve().parents[1] / "feedbax/contracts/stored_defaults_contract.json"
    assert canonical_json_v2_bytes(stored_defaults_manifest()) == canonical_json_v2_bytes(json.loads(path.read_bytes()))


def test_a_based_figure_writes_one_layer_edit_and_keeps_member_order(tmp_path):
    schema = STORED_FIGURE_DOCUMENT_SCHEMA_ID
    keyed = partial(stored_keyed_collection, schema)
    base = stored_project_value(StoredFigureDocument.model_validate(_FIGURE))
    assert isinstance(base, dict)
    target = deepcopy(base)
    target["layers"]["bounds"]["encode"]["upper"] = "upper95"
    changes = diff(base, target, keyed_collection=keyed)
    assert list(changes) == ["/layers/bounds/encode/upper"]
    assert changes["/layers/bounds/encode/upper"].value == "upper95"
    reordered = base | {"layers": dict(reversed(list(base["layers"].items())))}
    ordering = diff(base, reordered, keyed_collection=keyed)
    assert list(ordering) == ["/layers"]
    assert ordering["/layers"].order == ["zero", "bounds", "mean"]

    later = deepcopy(base)
    later["layout"]["markers"]["title"] = "Crossings"
    later["layers"]["mean"]["encode"]["y"] = "average"
    for index in range(20):
        later["layers"][f"marker-{index}"] = deepcopy(later["layers"]["zero"])
    widened = deepcopy(later)
    widened["layers"]["bounds"]["encode"]["upper"] = "upper95"
    assert diff(later, widened, keyed_collection=keyed) == changes
    ordered = apply_differences(later, ordering, keyed_collection=keyed)
    assert list(ordered["layers"]) == ["zero", "bounds", "mean", *(f"marker-{index}" for index in range(20))]

    project = Project(documents={
        "reach-velocity": RootDocument(schema=schema, label="Reach velocity", payload=later | {"theme": "plotly_white"}),
        "with-bounds": BasedDocument(schema=schema, label="With bounds", base={"document": "reach-velocity"},
                                    differences=changes),
    })

    def observe(project):
        composed = compose_document(project.documents.get, "with-bounds")
        if isinstance(composed, ProjectDiagnostic):
            return composed
        return StoredFigureDocument.model_validate(composed.payload).lower()

    files = _verify_parsimony(tmp_path, project, observe=observe)
    assert "theme" not in json.loads(files["documents/reach-velocity.json"])["payload"]
    assert json.loads(files["documents/with-bounds.json"])["differences"] == {
        "/layers/bounds/encode/upper": {"value": "upper95"},
    }
    figure = observe(project)
    assert [panel.name for panel in figure.layout] == ["velocity", "markers"]
    assert [layer.name for layer in figure.layers] == [*later["layers"]]
    assert figure.layout[1].title == "Crossings"
    assert figure.layers[0].encode.y == "average"
    assert figure.layers[1].encode.upper == "upper95"


@pytest.mark.parametrize(("old", "current"), [
    ("feedbax.figure_document.v1", STORED_FIGURE_DOCUMENT_SCHEMA_ID),
    ("feedbax.analysis_document.v1", ANALYSIS_SCHEMA),
])
def test_rejects_obsolete_authored_figure_and_analysis_versions(old, current):
    with pytest.raises(ValidationError, match=f"Expected {current!r}"):
        RootDocument(schema=old, label="Obsolete", payload={})


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

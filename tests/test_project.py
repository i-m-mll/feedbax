from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.objective_document import STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID
from feedbax.contracts.project import Project, RootDocument
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    ProjectReadError,
    StudioProjectFile,
    dump_studio_project,
    dump_text_project,
    load_text_project,
    parse_studio_project,
)
from feedbax.contracts.project_resolution import (
    BaseReaders,
    ProjectDiagnostic,
    ResolvedStep,
    ResolutionCode,
    SelectedRow,
    compose_step,
    resolve_step,
    validate_project_documents,
)
from feedbax.contracts.project_sweep import expand_sweep
from feedbax.contracts.training_document import TRAINING_DOCUMENT_SCHEMA_ID


def _write(root: Path, files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)


def _training_document(label: str, settings: dict) -> dict:
    return {"label": label, "schema": TRAINING_DOCUMENT_SCHEMA_ID, "payload": {
        "method": "feedbax.example_training", "settings": settings, "training_config": {},
    }}


def test_both_project_readers_round_trip_and_resolve(tmp_path):
    repeat = {"clauses": [{"attempt": [{"id": "high_repeat", "label": "High repeated"}]}]}
    high_seed = {"seed": {"value": 61}}
    repeated_reach = {"reach": {"value": 2.0}}
    graph = _training_document("Model", {"units": 48})
    project = Project.model_validate(
        {
            "documents": {
                "model": graph,
                "task": _training_document("Task", {"limits": {"reach/a": 1.0}}),
            },
            "steps": {
                "train": {
                    "label": "Train",
                    "kind": "train",
                    "base": {
                        "inline": {
                            "graph": {"document": "model"},
                            "reach": {"document": "task", "pointer": "/settings/limits/reach~1a"},
                            "seed": 17,
                        }
                    },
                    "requires": ["graph.settings.units", "reach"],
                    "sweep": {
                        "axes": {"attempt": {"values": {
                            "high": {"patches": high_seed},
                            "high_repeat": {"label": "High repeated", "patches": high_seed | repeated_reach},
                        }}},
                        "rows": {
                            "high": {"patches": high_seed},
                            "high_repeat": {"from": "high", "patches": repeated_reach},
                        },
                    },
                },
                "resume": {"label": "Resume", "kind": "train", "base": {"step": "train", "row": "high_repeat"}},
                "evaluate": {
                    "label": "Evaluate",
                    "kind": "evaluate",
                    "base": {"inline": {"condition": 0}},
                    "inputs": {"subject": {"step": "train", "where": repeat}},
                    "sweep": {
                        "axes": {
                            "condition": {
                                "values": {
                                    "low": {"patches": {"condition": {"value": 1}}},
                                    "high": {"patches": {"condition": {"value": 2}}},
                                }
                            }
                        }
                    },
                },
                "evaluate_subset": {
                    "label": "Evaluate subset",
                    "kind": "evaluate",
                    "base": {"step": "evaluate"},
                    "sweep": {
                        "axes": {
                            "condition": {
                                "values": {
                                    "low": None,
                                    "high": {"patches": {"condition": {"value": 4}}},
                                }
                            }
                        }
                    },
                },
            },
            "sweep": {
                "rows": {
                    "wide": {"patches": {"documents.model.payload.settings.units": {"value": 64}}},
                    "wide_repeat": {
                        "from": "wide",
                        "patches": {
                            "steps.train.base.inline.seed": {"value": 47},
                        },
                    },
                }
            },
        }
    )
    configuration = ProjectConfiguration(
        project="study", store="local/store", settings={"site": "lab"}
    )
    _write(tmp_path, dump_text_project(project, configuration))
    text = load_text_project(tmp_path)
    studio = parse_studio_project(
        dump_studio_project(
            StudioProjectFile(project=project, presentation={"viewport": [0, 0, 1]}),
        )
    )
    assert text.project == project == studio.project
    assert text.configuration == configuration
    assert text.store_path == tmp_path / "local/store"
    assert text.project.sweep is not None
    assert list(text.project.sweep.rows) == ["wide", "wide_repeat"]
    for name in project.steps:
        assert resolve_step(text.project, name) == resolve_step(studio.project, name)

    resume = resolve_step(text.project, "resume")
    assert isinstance(resume, ResolvedStep)
    assert resume.payload == {"graph": graph["payload"], "reach": 2.0, "seed": 61}
    assert resume.sweep is None
    evaluate = resolve_step(text.project, "evaluate_subset")
    assert isinstance(evaluate, ResolvedStep)
    assert evaluate.inputs["subject"].model_dump(exclude_unset=True) == {"step": "train", "where": repeat}
    assert evaluate.sweep is not None
    assert evaluate.sweep.model_dump(exclude_none=True)["axes"] == {
        "condition": {
            "values": {"high": {"patches": {"condition": {"op": "replace", "value": 4}}}}
        },
    }


def test_delta_diagnostics_and_document_reference_atomicity():
    project = Project.model_validate(
        {
            "documents": {"condition": _training_document("Condition", {"gain": 4})},
            "steps": {
                "base": {
                    "label": "Base",
                    "kind": "evaluate",
                    "base": {"inline": {"condition": {"gain": 0}}},
                    "deltas": [
                        {
                            "layer_id": "add-cue",
                            "patches": [
                                {"path": "condition.cue", "op": "add", "value": "go"},
                            ],
                        }
                    ],
                },
                "override": {
                    "label": "Override",
                    "kind": "evaluate",
                    "base": {"step": "base"},
                    "deltas": [
                        {
                            "layer_id": "change-cue",
                            "patches": [
                                {"path": "condition.cue", "value": "stop"},
                            ],
                        }
                    ],
                },
                "condition": {
                    "label": "Condition",
                    "kind": "evaluate",
                    "base": {"step": "base"},
                    "deltas": [
                        {
                            "layer_id": "bind-condition",
                            "acknowledges_ancestor_paths": ["condition"],
                            "patches": [
                                {"path": "condition", "value": {"document": "condition", "pointer": "/settings"}},
                            ],
                        }
                    ],
                },
                "bad_pointer_edit": {
                    "label": "Bad pointer edit",
                    "kind": "evaluate",
                    "base": {"step": "condition"},
                    "deltas": [
                        {
                            "layer_id": "edit-pointer",
                            "acknowledges_ancestor_paths": ["condition"],
                            "patches": [{"path": "condition.cue", "op": "add", "value": "wait"}],
                        }
                    ],
                },
            },
        }
    )
    override = resolve_step(project, "override")
    assert isinstance(override, ProjectDiagnostic)
    assert (override.code, override.step, override.layer, override.path) == (
        ResolutionCode.INVALID_DELTA,
        "override",
        "change-cue",
        "condition.cue",
    )
    reference = resolve_step(project, "condition")
    assert isinstance(reference, ResolvedStep)
    assert reference.payload == {"condition": {"gain": 4}}
    bad_pointer_edit = resolve_step(project, "bad_pointer_edit")
    assert isinstance(bad_pointer_edit, ProjectDiagnostic)
    assert (bad_pointer_edit.code, bad_pointer_edit.layer, bad_pointer_edit.path) == (
        ResolutionCode.INVALID_DELTA,
        "edit-pointer",
        "condition.cue",
    )


@pytest.mark.parametrize(
    ("documents", "steps", "code", "path"),
    [
        (
            {},
            {
                "a": {"label": "A", "kind": "train", "base": {"step": "b"}},
                "b": {"label": "B", "kind": "train", "base": {"step": "a"}},
            },
            ResolutionCode.STEP_CYCLE,
            "base.step",
        ),
        (
            {},
            {"a": {"label": "A", "kind": "train", "base": {"inline": {}}, "requires": ["task.rate"]}},
            ResolutionCode.UNBOUND_REQUIRED,
            "task.rate",
        ),
        (
            {"loop": _training_document("Loop", {"item": {"document": "loop", "pointer": "/settings"}})},
            {"a": {
                "label": "A", "kind": "train",
                "base": {"inline": {"task": {"document": "loop", "pointer": "/settings"}}},
            }},
            ResolutionCode.DOCUMENT_CYCLE,
            "payload.task.item",
        ),
        (
            {"task": _training_document("Task", {"items": []})},
            {"a": {"label": "A", "kind": "train", "base": {"document": "task", "pointer": "/settings/items/0"}}},
            ResolutionCode.BAD_POINTER,
            "base",
        ),
    ],
)
def test_resolution_failure_names_the_location(documents, steps, code, path):
    failure = resolve_step(Project.model_validate({"documents": documents, "steps": steps}), "a")
    assert isinstance(failure, ProjectDiagnostic)
    assert (failure.code, failure.path) == (code, path)


def test_row_sources_survive_authored_only_and_generated_rows_use_exact_reader():
    project = Project.model_validate(
        {
            "steps": {
                "first": {
                    "label": "First",
                    "kind": "train",
                    "base": {"inline": {"seed": 0, "batches": 2}},
                    "sweep": {"rows": {"original": {"patches": {"seed": {"value": 19}}}}},
                },
                "derived": {
                    "label": "Derived",
                    "kind": "train",
                    "base": {"step": "first"},
                    "sweep": {
                        "inherit_rows": False,
                        "rows": {
                            "long": {
                                "from": "original",
                                "patches": {
                                    "batches": {"value": 200},
                                    "model_specific_delta": {"op": "add", "value": {"floor": 2}},
                                },
                            },
                            "long_repeat": {
                                "from": "long",
                                "patches": {
                                    "seed": {"value": 20},
                                    "model_specific_delta": {
                                        "op": "replace",
                                        "value": {"floor": 4},
                                    },
                                },
                            },
                        },
                    },
                },
                "from_local": {"label": "From local", "kind": "train", "base": {"step": "derived", "row": "long_repeat"}},
                "from_generated": {
                    "label": "From generated",
                    "kind": "train",
                    "base": {"step": "first", "row": "seed=sample"},
                },
            }
        }
    )
    resolved = resolve_step(project, "derived")
    assert isinstance(resolved, ResolvedStep)
    assert resolved.sweep is not None
    assert list(resolved.sweep.rows) == ["long", "long_repeat"]
    local = resolve_step(project, "from_local")
    assert isinstance(local, ResolvedStep)
    assert local.payload == {"seed": 20, "batches": 200, "model_specific_delta": {"floor": 4}}
    assert local.sweep is None

    def select(step, row):
        assert step.name == "first"
        assert row == "seed=sample"
        return SelectedRow(step.payload | {"seed": 83}, step.inputs)

    generated = resolve_step(project, "from_generated", readers=BaseReaders(row=select))
    assert isinstance(generated, ResolvedStep)
    assert generated.payload == {"seed": 83, "batches": 2}
    assert generated.sweep is None


def test_external_and_historical_bases_use_exact_readers():
    raw = b'{"old":{"training":{"batches":112}}}'
    project = Project.model_validate(
        {
            "steps": {
                "historical": {"label": "Historical", "kind": "train", "base": {"run": "recorded-request"}},
                "external": {
                    "label": "External",
                    "kind": "train",
                    "base": {
                        "path": "../record.json",
                        "sha256": sha256_bytes(raw),
                        "pointer": "/old",
                    },
                },
            }
        }
    )
    data = {"training": {"batches": 112}}
    recorded = data | {"opaque": {"document": "recorded-literal"}}
    readers = BaseReaders(
        run=lambda identifier: {"recorded-request": recorded}.get(identifier),
        external=lambda path: {"../record.json": raw}.get(path),
    )
    for name in project.steps:
        resolved = resolve_step(project, name, readers=readers)
        assert isinstance(resolved, ResolvedStep)
        assert resolved.payload == (recorded if name == "historical" else data)

    def unreadable(path):
        raise PermissionError(path)

    failure = resolve_step(project, "external", readers=BaseReaders(external=unreadable))
    assert isinstance(failure, ProjectDiagnostic)
    assert (failure.code, failure.step, failure.path) == (
        ResolutionCode.MISSING_AUTHORITY,
        "external",
        "base.path",
    )

    mismatch = resolve_step(project, "external", readers=BaseReaders(external=lambda _: b"{}"))
    assert isinstance(mismatch, ProjectDiagnostic)
    assert mismatch.code == ResolutionCode.BAD_EXTERNAL_CONTENT
    assert mismatch.path == "base.sha256"


def test_document_schema_validation_distinguishes_unknown_schema_reference_and_value():
    class Rates(BaseModel):
        rate: int

    class Training(BaseModel):
        settings: Rates

    project = Project.model_validate({"documents": {
        "good": _training_document("Good", {"rate": 1}),
        "bad": _training_document("Bad", {"rate": "never"}),
        "unrecognized": {
            "label": "Unrecognized", "schema": STORED_OBJECTIVE_DOCUMENT_SCHEMA_ID, "payload": {"id": "objective"},
        },
        "unresolved": _training_document("Unresolved", {"document": "missing"}),
    }})
    errors = validate_project_documents(project, {TRAINING_DOCUMENT_SCHEMA_ID: Training.model_validate})
    assert [(item.code, item.path) for item in errors] == [
        (ResolutionCode.INVALID_DOCUMENT, "payload.settings.rate"),
        (ResolutionCode.UNKNOWN_SCHEMA, "schema"),
        (ResolutionCode.MISSING_DOCUMENT, "payload.settings"),
    ]


def test_project_documents_refuse_unregistered_storage_schema():
    with pytest.raises(ValidationError, match="No registered stored document has schema 'unknown'"):
        RootDocument(label="Unknown", schema="unknown", payload={})


def test_reader_refuses_duplicate_json_members():
    with pytest.raises(ProjectReadError, match="duplicate.*project"):
        parse_studio_project(b'{"project":{},"project":{},"presentation":{}}')


def test_document_base_follows_root_alias_before_deltas():
    project = Project.model_validate(
        {
            "documents": {
                "condition": _training_document("Condition", {"gain": 3}),
                "alias": _training_document("Alias", {
                    "target": {"document": "condition", "pointer": "/settings"},
                }),
            },
            "steps": {
                "evaluate": {
                    "label": "Evaluate",
                    "kind": "evaluate",
                    "base": {"document": "alias", "pointer": "/settings/target"},
                    "deltas": [
                        {"layer_id": "higher-gain", "patches": [{"path": "gain", "value": 6}]}
                    ],
                }
            },
        }
    )
    resolved = resolve_step(project, "evaluate")
    assert isinstance(resolved, ResolvedStep)
    assert resolved.payload == {"gain": 6}
    assert project.documents["condition"].payload["settings"] == {"gain": 3}


def test_selected_row_sources_follow_expression_edits_inherited_rows_and_document_paths():
    from feedbax.contracts.task_document import STORED_TASK_DOCUMENT_SCHEMA_ID

    project = Project.model_validate(
        {
            "documents": {
                "task": {
                    "schema": STORED_TASK_DOCUMENT_SCHEMA_ID,
                    "label": "Task",
                    "payload": {
                        "id": "task",
                        "variables": {"goal": {
                            "label": "Goal", "type": {"base": "real", "shape": [3]},
                            "value": "vector(2, 4)", "count": 10,
                        }},
                        "epochs": {"trial": {"label": "Trial", "end": {"kind": "length", "length": 1}}},
                        "signals": {},
                    },
                },
            },
            "steps": {
                "fit": {
                    "label": "Fit",
                    "kind": "train",
                    "base": {"document": "task", "pointer": "/variables/goal"},
                    "deltas": [
                        {
                            "layer_id": "insert",
                            "patches": [{
                                "path": "value.args.0", "op": "add", "value": 1,
                            }],
                        },
                    ],
                    "sweep": {
                        "rows": {
                            "first": {"patches": {"count": {"value": 11}}},
                            "derived": {"from": "first"},
                        },
                        "exceptions": {
                            "raise": {
                                "where": {"rows": ["derived"]},
                                "patches": {"count": {"value": 12}},
                            },
                        },
                    },
                },
                "child": {"label": "Child", "kind": "train", "base": {"step": "fit", "row": "derived"}},
            },
        }
    )
    template = compose_step(project, "child", track_sources=True)
    assert not isinstance(template, ProjectDiagnostic)
    result = template.resolve(include_sources=True)
    assert isinstance(result, ResolvedStep)
    assert result.payload["value"] == "[1, 2, 4]"
    assert result.payload["count"] == 12
    assert result.sources is not None
    assert result.sources["/value/args/0/value"].path == "/deltas/0/patches/0/value"
    assert result.sources["/value/args/1/value"].path == "/variables/goal/value"
    assert result.sources["/value/args/2/value"].path == "/variables/goal/value"
    assert (result.sources["/count"].kind, result.sources["/count"].name) == ("exception", "raise")


def test_selected_sibling_rows_keep_original_sources_after_member_writes():
    project = Project.model_validate({
        "documents": {"shared": _training_document("Shared", {"a/b": {"~key": 5}})},
        "steps": {"fit": {
            "label": "Fit",
            "kind": "train",
            "base": {"inline": {"items": {
                "shared": {"document": "shared", "pointer": "/settings/a~1b"}, "native": [],
            }}},
            "sweep": {"rows": {
                "inserted": {"patches": {"items.inserted": {"op": "add", "value": 7}}},
                "removed": {"patches": {"items.shared": {"op": "remove"}}},
                "original": {},
            }},
        }},
    })
    template = compose_step(project, "fit", track_sources=True)
    assert not isinstance(template, ProjectDiagnostic)
    inserted, removed, original = (
        template.resolve(row, include_sources=True) for row in expand_sweep(template.sweep)
    )
    assert isinstance(inserted, ResolvedStep) and inserted.sources is not None
    assert isinstance(removed, ResolvedStep) and removed.sources is not None
    assert isinstance(original, ResolvedStep) and original.sources is not None
    assert list(inserted.payload["items"].items()) == [("shared", {"~key": 5}), ("native", []), ("inserted", 7)]
    assert removed.payload["items"] == {"native": []}
    assert original.payload["items"] == {"shared": {"~key": 5}, "native": []}
    assert inserted.sources["/items/inserted"].path == "/sweep/rows/inserted/patches/items.inserted/value"
    assert inserted.sources["/items/shared/~0key"] == original.sources["/items/shared/~0key"]
    assert original.sources["/items/shared/~0key"].path == "/settings/a~1b/~0key"
    assert removed.sources["/items/native"] == inserted.sources["/items/native"]
    assert removed.sources["/items/native"].path == "/base/inline/items/native"


def test_sweep_routes_reject_arrays_and_remove_absent_fields_in_referenced_documents():
    project = Project.model_validate({
        "documents": {
            "shared": _training_document("Shared", {"weights": [1], "amount": 2}),
        },
        "steps": {"fit": {
            "label": "Fit",
            "kind": "train",
            "base": {"inline": {
                "settings": {"document": "shared", "pointer": "/settings"},
                "aliases": [{"id": "first", "ref": {"document": "shared", "pointer": "/settings"}}],
            }},
            "sweep": {"rows": {
                "remove-absent": {"patches": {"settings.defaulted": {"op": "remove"}}},
                "array-in-document": {"patches": {"settings.weights.0": {"value": 3}}},
                "array-before-reference": {"patches": {"aliases.first.ref.amount": {"value": 3}}},
            }},
        }},
    })
    template = compose_step(project, "fit", track_sources=True)
    assert not isinstance(template, ProjectDiagnostic)
    resolved, indexed, redirected = (
        template.resolve(row, include_sources=True) for row in expand_sweep(template.sweep)
    )

    assert isinstance(resolved, ResolvedStep) and resolved.sources is not None
    assert resolved.payload["settings"] == {"weights": [1], "amount": 2}
    assert resolved.sources["/settings/amount"].name == "shared"
    for problem in (indexed, redirected):
        assert isinstance(problem, ProjectDiagnostic) and problem.code == ResolutionCode.INVALID_SWEEP
        assert "cannot traverse an array" in problem.detail

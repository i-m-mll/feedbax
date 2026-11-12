from pathlib import Path

import pytest
from pydantic import BaseModel

from feedbax.contracts.base import sha256_bytes
from feedbax.contracts.project import Document, Project
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


def _write(root: Path, files: dict[str, bytes]) -> None:
    for name, payload in files.items():
        path = root / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(payload)


def test_both_project_readers_round_trip_and_resolve(tmp_path):
    project = Project.model_validate(
        {
            "documents": {
                "model": {"schema": "model.v1", "payload": {"units": 48}},
                "task": {"schema": "task.v1", "payload": {"limits": {"reach/a": 1.0}}},
            },
            "steps": {
                "train": {
                    "kind": "train",
                    "base": {
                        "inline": {
                            "graph": {"document": "model"},
                            "reach": {"document": "task", "pointer": "/limits/reach~1a"},
                            "seed": 17,
                        }
                    },
                    "requires": ["graph.units", "reach"],
                    "sweep": {
                        "rows": {
                            "high": {"patches": {"seed": {"value": 61}}},
                            "high_repeat": {"from": "high", "patches": {"reach": {"value": 2.0}}},
                        }
                    },
                },
                "resume": {"kind": "train", "base": {"step": "train", "row": "high_repeat"}},
                "evaluate": {
                    "kind": "evaluate",
                    "base": {"inline": {"condition": 0}},
                    "inputs": {"subject": {"step": "train", "row": "high_repeat"}},
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
                    "wide": {"patches": {"documents.model.payload.units": {"value": 64}}},
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
    assert resume.payload == {"graph": {"units": 48}, "reach": 2.0, "seed": 61}
    assert resume.sweep is None
    evaluate = resolve_step(text.project, "evaluate_subset")
    assert isinstance(evaluate, ResolvedStep)
    assert evaluate.inputs["subject"].model_dump(exclude_none=True) == {
        "step": "train",
        "row": "high_repeat",
    }
    assert evaluate.sweep is not None
    assert evaluate.sweep.model_dump(exclude_none=True)["axes"] == {
        "condition": {
            "values": {"high": {"patches": {"condition": {"op": "replace", "value": 4}}}}
        },
    }


def test_delta_diagnostics_and_document_reference_atomicity():
    project = Project.model_validate(
        {
            "documents": {"condition": {"schema": "condition.v1", "payload": {"gain": 4}}},
            "steps": {
                "base": {
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
                    "kind": "evaluate",
                    "base": {"step": "base"},
                    "deltas": [
                        {
                            "layer_id": "bind-condition",
                            "acknowledges_ancestor_paths": ["condition"],
                            "patches": [
                                {"path": "condition", "value": {"document": "condition"}},
                            ],
                        }
                    ],
                },
                "bad_pointer_edit": {
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
                "a": {"kind": "train", "base": {"step": "b"}},
                "b": {"kind": "train", "base": {"step": "a"}},
            },
            ResolutionCode.STEP_CYCLE,
            "base.step",
        ),
        (
            {},
            {"a": {"kind": "train", "base": {"inline": {}}, "requires": ["task.rate"]}},
            ResolutionCode.UNBOUND_REQUIRED,
            "task.rate",
        ),
        (
            {"loop": {"schema": "task", "payload": {"item": {"document": "loop"}}}},
            {"a": {"kind": "train", "base": {"inline": {"task": {"document": "loop"}}}}},
            ResolutionCode.DOCUMENT_CYCLE,
            "payload.task.item",
        ),
        (
            {"task": {"schema": "task", "payload": {"items": []}}},
            {"a": {"kind": "train", "base": {"document": "task", "pointer": "/items/0"}}},
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
                    "kind": "train",
                    "base": {"inline": {"seed": 0, "batches": 2}},
                    "sweep": {"rows": {"original": {"patches": {"seed": {"value": 19}}}}},
                },
                "derived": {
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
                "from_local": {"kind": "train", "base": {"step": "derived", "row": "long_repeat"}},
                "from_generated": {
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
                "historical": {"kind": "train", "base": {"run": "recorded-request"}},
                "external": {
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
    class Task(BaseModel):
        rate: int

    project = Project(
        documents={
            "good": Document(schema="task", payload={"rate": 1}),
            "bad": Document(schema="task", payload={"rate": "never"}),
            "unknown": Document(schema="unknown", payload={}),
            "unresolved": Document(schema="task", payload={"document": "missing"}),
        }
    )
    errors = validate_project_documents(project, {"task": Task.model_validate})
    assert [(item.code, item.path) for item in errors] == [
        (ResolutionCode.INVALID_DOCUMENT, "payload.rate"),
        (ResolutionCode.UNKNOWN_SCHEMA, "schema"),
        (ResolutionCode.MISSING_DOCUMENT, "payload"),
    ]


def test_reader_refuses_duplicate_json_members():
    with pytest.raises(ProjectReadError, match="duplicate.*project"):
        parse_studio_project(b'{"project":{},"project":{},"presentation":{}}')


def test_document_base_follows_root_alias_before_deltas():
    project = Project.model_validate(
        {
            "documents": {
                "condition": {"schema": "condition", "payload": {"gain": 3}},
                "alias": {"schema": "condition", "payload": {"document": "condition"}},
            },
            "steps": {
                "evaluate": {
                    "kind": "evaluate",
                    "base": {"document": "alias"},
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
    assert project.documents["condition"].payload == {"gain": 3}


def test_selected_row_sources_follow_list_edits_inherited_rows_and_document_paths():
    project = Project.model_validate(
        {
            "documents": {
                "settings": {"schema": "settings", "payload": {"values": [2, 4], "limit": 10}},
            },
            "steps": {
                "fit": {
                    "kind": "train",
                    "base": {"document": "settings"},
                    "deltas": [
                        {
                            "layer_id": "insert",
                            "patches": [{"path": "values.0", "op": "add", "value": 1}],
                        },
                    ],
                    "sweep": {
                        "rows": {
                            "first": {"patches": {"limit": {"value": 11}}},
                            "derived": {"from": "first"},
                        },
                        "exceptions": {
                            "raise": {
                                "where": {"rows": ["derived"]},
                                "patches": {"limit": {"value": 12}},
                            },
                        },
                    },
                },
                "child": {"kind": "train", "base": {"step": "fit", "row": "derived"}},
            },
        }
    )
    template = compose_step(project, "child", track_sources=True)
    assert not isinstance(template, ProjectDiagnostic)
    result = template.resolve(include_sources=True)
    assert isinstance(result, ResolvedStep)
    assert result.payload == {"values": [1, 2, 4], "limit": 12}
    assert result.sources is not None
    assert result.sources["/values/0"].path == "/deltas/0/patches/0/value"
    assert result.sources["/values/1"].path == "/values/0"
    assert result.sources["/values/2"].path == "/values/1"
    assert (result.sources["/limit"].kind, result.sources["/limit"].name) == ("exception", "raise")

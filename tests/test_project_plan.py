import json
from pathlib import Path

from feedbax.contracts.project import Project
from feedbax.contracts.project_inputs import SubsetTerms, subset_to_rows
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    StudioProjectFile,
    dump_studio_project,
    dump_text_project,
    load_text_project,
    parse_studio_project,
)
from feedbax.contracts.project_plan import ProjectPlanMemo, RunKey, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
from feedbax.contracts.project_sweep import (
    SweepInputMember,
    axis_to_list,
    expand_sweep,
    sweep_to_rows,
)
from feedbax.contracts.request_identity import decode_request
from feedbax.contracts.request_refs import BestMember, InputMember, LastMember, ProductRef


def _project_data():
    corpus = json.loads(
        (Path(__file__).parent / "fixtures" / "request_identity_corpus.json").read_bytes()
    )["requests"]
    train = next(row for row in corpus if row["kind"] == "train")
    evaluate = next(row for row in corpus if row["kind"] == "evaluate")
    train_spec = train["spec"]
    graph = train_spec.pop("graph")
    return (
        {
            "documents": {"model": {"schema": graph["schema_id"], "payload": graph}},
            "steps": {
                "fit": {
                    "kind": "train",
                    "base": {"inline": {"graph": {"document": "model"}, **train_spec}},
                },
                "evaluate": {
                    "kind": "evaluate",
                    "base": {"inline": evaluate["spec"]},
                    "inputs": {"subject": {"step": "fit", "checkpoint": {"best_by": "loss"}}},
                },
            },
        },
        train["id"],
    )


def test_text_and_studio_compile_one_shared_plan_before_any_results(tmp_path):
    data, expected_train = _project_data()
    project = Project.model_validate(data)
    files = dump_text_project(project, ProjectConfiguration(project="experiment", store="results"))
    for name, content in files.items():
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(content)
    text = load_text_project(tmp_path)
    studio = parse_studio_project(
        dump_studio_project(
            StudioProjectFile(project=project, presentation={"viewport": [0, 0, 1]})
        )
    )
    plan = compile_project(text.project)
    assert plan == compile_project(studio.project)
    assert not plan.diagnostics
    train, evaluation = plan.runs
    assert train.key == RunKey("default", "fit", "default")
    assert train.request_id == expected_train
    assert evaluation.key == RunKey("default", "evaluate", "default")
    assert evaluation.inputs["subject"].producer == train.key

    train_request = decode_request(plan.requests[train.request_id], plan.documents.__getitem__)
    evaluation_request = decode_request(
        plan.requests[evaluation.request_id], plan.documents.__getitem__
    )
    assert train_request.kind == "train" and train_request.inputs == {}
    assert evaluation_request.kind == "evaluate" and evaluation_request.id == evaluation.request_id
    assert evaluation_request.inputs == {
        "subject": (
            InputMember(
                key="default",
                reference=ProductRef(
                    request_id=train.request_id,
                    role="checkpoint",
                    select=BestMember(metric="loss"),
                ),
            ),
        ),
    }


def test_named_document_difference_is_local_and_updates_memoized_sources():
    data, _ = _project_data()
    data["steps"]["sibling"] = {"kind": "train", "base": {"step": "fit"}}
    data["steps"]["fit"]["sweep"] = {
        "axes": {"seed": {"values": {"fixed": {"patches": {"seed": {"value": 2}}}}}},
    }
    project = Project.model_validate(data)
    memo = ProjectPlanMemo()
    first = compile_project(project, memo=memo)
    fit = next(run for run in first.runs if run.key.step == "fit")
    sibling = next(run for run in first.runs if run.key.step == "sibling")
    keys = [fit.key, sibling.key]
    assert first.document_users("default", "model") == {"fit", "sibling"}

    path = "/nodes/gain/params/gain"
    data["steps"]["fit"]["document_overrides"] = {"model": {path: 3}}
    scoped = Project.model_validate(data)
    plan = compile_project(scoped, source_runs=keys, memo=memo)
    assert plan == compile_project(scoped, source_runs=keys)
    requests = {run.key: run.request_id for run in plan.runs}
    assert requests[fit.key] != fit.request_id
    assert requests[sibling.key] == sibling.request_id
    assert plan.document_users("default", "model") == {"fit", "sibling"}
    fit_source = plan.sources[fit.key]["/payload/graph" + path]
    assert (fit_source.kind, fit_source.name) == ("step", "fit")
    assert plan.sources[sibling.key]["/payload/graph" + path].name == "model"
    assert plan.selected_definitions[fit.key].payload["seed"] == 2

    data["documents"]["model"]["payload"]["nodes"]["gain"]["params"]["gain"] = 4
    del data["steps"]["fit"]["document_overrides"]
    shared = Project.model_validate(data)
    plan = compile_project(shared, source_runs=keys, memo=memo)
    assert plan == compile_project(shared, source_runs=keys)
    assert plan.sources[fit.key]["/payload/graph" + path].name == "model"


def test_fanout_selects_rows_and_requires_a_valid_static_subject():
    data, _ = _project_data()
    fit = data["steps"]["fit"]
    fit["requires"] = ["seed"]
    fit["base"]["inline"]["seed"] = None
    fit["sweep"] = {
        "axes": {
            "seed": {
                "values": {
                    "a": {"patches": {"seed": {"value": 1}}},
                    "b": {"patches": {"seed": {"value": 2}}},
                }
            }
        }
    }
    evaluation = data["steps"]["evaluate"]
    ambiguous = compile_project(Project.model_validate(data))
    [problem] = ambiguous.diagnostics
    assert (problem.step, problem.row, problem.path) == ("evaluate", "default", "inputs.subject")
    assert "seed=a" in problem.detail and "seed=b" in problem.detail

    evaluation["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    evaluation["inputs"]["subject"]["checkpoint"] = "last"
    plan = compile_project(Project.model_validate(data))
    assert not plan.diagnostics
    assert len(plan.runs) == len(plan.requests) == 4
    producers = {run.key: run for run in plan.runs if run.kind == "train"}
    for run in (run for run in plan.runs if run.kind == "evaluate"):
        selected = run.inputs["subject"]
        producer = producers[selected.producer]
        assert run.coordinates["seed"] == producer.coordinates["seed"]
        assert selected.member.reference == ProductRef(
            request_id=producer.request_id, role="checkpoint", select=LastMember()
        )
    assert {run.requires for run in producers.values()} == {("seed",)}


def _chained_train_data():
    data, _ = _project_data()
    steps = data["steps"]
    del steps["evaluate"]
    steps["fit"]["sweep"] = {
        "axes": {
            "seed": {
                "values": {
                    "a": {"label": "Replicate", "patches": {"seed": {"value": 1}}},
                    "b": {"label": "Replicate", "patches": {"seed": {"value": 2}}},
                }
            }
        },
        "rows": {"manual": {"patches": {"seed": {"value": 99}}}},
    }
    for name in ("flat", "rewarm"):
        steps[name] = {
            "kind": "train",
            "base": {"step": "fit"},
            "inputs": {"checkpoint": {"step": "fit", "checkpoint": "last"}},
        }
    steps["flat"]["sweep"] = {
        "axes": {
            "rate": {
                "values": {
                    "fast": {"patches": {"training_config.learning_rate": {"value": 0.001}}},
                    "slow": {"patches": {"training_config.learning_rate": {"value": 0.0001}}},
                }
            }
        }
    }
    steps["rewarm"]["deltas"] = [
        {"layer_id": "rewarm", "patches": [{"path": "training_config.n_batches", "value": 50}]}
    ]
    return data


def test_chained_train_pairs_named_and_generated_rows_before_and_after_suffix_edit():
    data = _chained_train_data()
    data["sweep"] = {
        "rows": {
            "baseline": {},
            "alternate": {
                "patches": {
                    "steps.fit.base.inline.training_config.n_batches": {"value": 40}
                }
            },
        }
    }
    memo = ProjectPlanMemo()
    selected = RunKey("alternate", "flat", "rate=fast--seed=b")
    before = compile_project(Project.model_validate(data), memo=memo, source_runs=[selected])
    assert not before.diagnostics
    runs = {run.key: run for run in before.runs}
    assert len(runs) == 2 * (3 + 5 + 3)
    for run in runs.values():
        if run.key.step == "fit":
            assert not run.inputs
            continue
        parent_row = f"seed={run.coordinates['seed'].value_id}" if run.coordinates else "manual"
        parent = runs[RunKey(run.key.variant, "fit", parent_row)]
        checkpoint = run.inputs["checkpoint"]
        assert checkpoint.producer == parent.key
        request = decode_request(before.requests[run.request_id], before.documents.__getitem__)
        assert request.inputs == {
            "checkpoint": (
                InputMember(
                    key=parent_row,
                    reference=ProductRef(
                        request_id=parent.request_id, role="checkpoint", select=LastMember()
                    ),
                ),
            )
        }
    source = before.sources[selected]["/inputs/checkpoint"]
    assert (source.step, source.variant, source.row) == ("fit", "alternate", "seed=b")

    data["steps"]["flat"]["sweep"]["exceptions"] = {
        "extend": {
            "where": {"axes": {"seed": ["b"], "rate": ["fast"]}},
            "patches": {"training_config.n_batches": {"value": 80}},
        }
    }
    edited = Project.model_validate(data)
    after = compile_project(edited, memo=memo, source_runs=[selected])
    assert after == compile_project(edited, source_runs=[selected])
    assert not after.diagnostics
    assert {run.key for run in after.runs} == set(runs)
    assert {
        run.key for run in after.runs if run.request_id != runs[run.key].request_id
    } == {RunKey(variant, "flat", selected.row) for variant in before.variants}


def test_chained_train_requires_a_row_match_or_explicit_checkpoint_mapping():
    data = _chained_train_data()
    flat = data["steps"]["flat"]
    flat["sweep"]["axes"]["seed"] = None
    flat["sweep"]["inherit_rows"] = False
    plan = compile_project(Project.model_validate(data))
    assert {
        (problem.step, problem.variant, problem.row, problem.path) for problem in plan.diagnostics
    } == {
        ("flat", "default", f"rate={rate}", "inputs.checkpoint") for rate in ("fast", "slow")
    }
    assert all(problem.code == ResolutionCode.BAD_REFERENCE for problem in plan.diagnostics)
    assert all("manual, seed=a, seed=b" in problem.detail for problem in plan.diagnostics)
    assert all(run.key.step != "flat" for run in plan.runs)

    flat["inputs"]["checkpoint"]["where"] = {"rows": ["seed=b", "manual"]}
    flat["sweep"]["axes"]["parent"] = {"input": "checkpoint"}
    mapped = compile_project(Project.model_validate(data))
    assert not mapped.diagnostics
    parents = [
        run.inputs["checkpoint"].producer.row for run in mapped.runs if run.key.step == "flat"
    ]
    assert sorted(parents) == ["manual", "manual", "seed=b", "seed=b"]


def test_checkpoint_subset_pairs_surviving_ids_even_with_one_candidate():
    data = _chained_train_data()
    data["steps"]["rewarm"]["base"]["step"] = "flat"
    data["steps"]["rewarm"]["inputs"]["checkpoint"]["step"] = "flat"
    checkpoint = data["steps"]["flat"]["inputs"]["checkpoint"]
    checkpoint["where"] = {"axes": {"seed": ["b"]}}
    memo = ProjectPlanMemo()
    plan = compile_project(Project.model_validate(data), memo=memo)
    assert not plan.diagnostics
    flat = [run for run in plan.runs if run.key.step == "flat"]
    assert {run.key.row for run in flat} == {"rate=fast--seed=b", "rate=slow--seed=b"}
    assert [row for row in plan.rows if row.key.step == "flat"] == flat
    assert all(run.inputs["checkpoint"].producer.row == "seed=b" for run in flat)
    rewarm = [run for run in plan.runs if run.key.step == "rewarm"]
    assert [row for row in plan.rows if row.key.step == "rewarm"] == rewarm
    assert [run.inputs["checkpoint"].producer for run in rewarm] == [run.key for run in flat]

    del checkpoint["where"]
    checkpoint["row"] = "seed=b"
    explicit = compile_project(Project.model_validate(data), memo=memo)
    assert explicit == compile_project(Project.model_validate(data))
    assert not explicit.diagnostics
    flat = [run for run in explicit.runs if run.key.step == "flat"]
    assert len(flat) == 5
    assert all(run.inputs["checkpoint"].producer.row == "seed=b" for run in flat)


def test_only_an_unswept_checkpoint_defaults_across_suffix_local_axes():
    data = _chained_train_data()
    prefix = data["steps"]["fit"]
    del prefix["sweep"]["rows"]
    del prefix["sweep"]["axes"]["seed"]["values"]["b"]
    data["steps"]["flat"]["sweep"]["axes"]["seed"] = {
        "values": {"a": None, "new": {"patches": {"seed": {"value": 2}}}}
    }
    plan = compile_project(Project.model_validate(data))
    assert {problem.row for problem in plan.diagnostics} == {
        "rate=fast--seed=new", "rate=slow--seed=new"
    }
    assert all(
        problem.step == "flat" and "Candidates: seed=a." in problem.detail
        for problem in plan.diagnostics
    )
    assert all(run.key.step != "flat" for run in plan.runs)

    del prefix["sweep"]
    del data["steps"]["flat"]["sweep"]["axes"]["seed"]
    default = compile_project(Project.model_validate(data))
    assert not default.diagnostics
    flat = [run for run in default.runs if run.key.step == "flat"]
    assert {run.key.row for run in flat} == {"rate=fast", "rate=slow"}
    assert all(run.inputs["checkpoint"].producer == RunKey("default", "fit", "default") for run in flat)


def test_invalid_row_is_diagnosed_without_rebinding_the_consumer():
    data, _ = _project_data()
    data["steps"]["fit"]["sweep"] = {
        "rows": {
            "good": {},
            "bad": {"patches": {"seed": {"value": "text"}}},
        }
    }
    plan = compile_project(Project.model_validate(data))
    assert [run.key for run in plan.runs] == [RunKey("default", "fit", "good")]
    invalid, dependent = plan.diagnostics
    assert (invalid.code, invalid.step, invalid.row, invalid.path) == (
        ResolutionCode.INVALID_PAYLOAD,
        "fit",
        "bad",
        "payload.seed",
    )
    assert (dependent.step, dependent.path) == ("evaluate", "inputs.subject")


def test_duplicate_specs_share_one_request_without_losing_run_keys():
    data, expected_train = _project_data()
    del data["steps"]["evaluate"]
    data["steps"]["fit"]["sweep"] = {
        "rows": {"repeat": {"label": "Again"}, "original": {"label": "First"}}
    }
    plan = compile_project(Project.model_validate(data))
    assert not plan.diagnostics
    assert [run.label for run in plan.runs] == ["Again", "First"]
    assert {run.key.row for run in plan.runs} == {"repeat", "original"}
    assert set(plan.requests) == {expected_train}


def test_variants_keep_run_keys_separate_and_deduplicate_unchanged_requests():
    data, expected_train = _project_data()
    data["sweep"] = {
        "rows": {
            "base": {},
            "copy": {},
            "seed8": {"patches": {"steps.fit.base.inline.seed": {"value": 8}}},
        }
    }
    data["steps"]["evaluate"]["inputs"]["subject"]["checkpoint"] = "last"
    project = Project.model_validate(data)
    requested = RunKey("seed8", "fit", "default")
    plan = compile_project(project, source_runs=[requested])
    assert not plan.diagnostics
    runs = {run.key: run for run in plan.runs}
    assert len(runs) == 6
    assert len(plan.requests) == 4
    assert runs[RunKey("base", "fit", "default")].request_id == expected_train
    assert runs[RunKey("copy", "fit", "default")].request_id == expected_train
    for variant in ("base", "copy", "seed8"):
        evaluation = runs[RunKey(variant, "evaluate", "default")]
        assert evaluation.inputs["subject"].producer == RunKey(variant, "fit", "default")
    assert set(plan.sources) == {requested}
    seed = plan.sources[requested]["/payload/seed"]
    assert (seed.kind, seed.variant, seed.path) == (
        "variant",
        "seed8",
        "/steps/fit/base/inline/seed",
    )
    assert compile_project(project).sources == {}

    data["steps"]["evaluate"]["inputs"]["subject"]["variants"] = ["base", "seed8"]
    data["steps"]["evaluate"]["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    cross = compile_project(Project.model_validate(data))
    assert not cross.diagnostics
    for variant in ("base", "copy", "seed8"):
        readers = [
            run for run in cross.runs if (run.key.variant, run.kind) == (variant, "evaluate")
        ]
        assert [reader.inputs["subject"].member.key for reader in readers] == [
            "base/default",
            "seed8/default",
        ]
    project = Project.model_validate(data)
    evaluation = project.steps["evaluate"]
    evaluation.sweep = sweep_to_rows(
        evaluation.sweep,
        inputs={"subject": [SweepInputMember("base/default"), SweepInputMember("seed8/default")]},
    )
    reopened = parse_studio_project(
        dump_studio_project(StudioProjectFile(project=project, presentation={}))
    )
    assert compile_project(reopened.project) == cross


def test_generated_variant_ids_can_be_named_in_references():
    data, _ = _project_data()
    data["sweep"] = {"axes": {"seed": {"path": "steps.fit.base.inline.seed", "values": [1, 2]}}}
    project = Project.model_validate(data)
    first, second = expand_sweep(project.sweep)
    data["steps"]["evaluate"]["inputs"]["subject"]["variants"] = [first.id]
    plan = compile_project(Project.model_validate(data))
    assert not plan.diagnostics
    [reader] = [run for run in plan.runs if run.key == RunKey(second.id, "evaluate", "default")]
    assert reader.inputs["subject"].producer.variant == first.id


def test_selection_by_axis_or_exact_rows_binds_statically_in_upstream_order():
    data, _ = _project_data()
    data["steps"]["fit"]["sweep"] = {
        "axes": {
            "seed": {
                "values": {
                    name: {"patches": {"seed": {"value": value}}}
                    for name, value in (("a", 1), ("b", 2), ("c", 3))
                }
            }
        }
    }
    evaluation = data["steps"]["evaluate"]
    evaluation["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    reference = evaluation["inputs"]["subject"]
    reference["where"] = {"axes": {"seed": ["a", "c"]}, "rows": ["seed=c", "seed=a"]}
    plan = compile_project(Project.model_validate(data))
    assert not plan.diagnostics
    evaluation_runs = [run for run in plan.runs if run.kind == "evaluate"]
    assert [run.coordinates["seed"].value_id for run in evaluation_runs] == ["a", "c"]
    subset = plan.subsets["default", "evaluate", "subject"]
    assert subset.candidates == 3
    assert tuple(key.row for key in subset.members) == ("seed=a", "seed=c")

    data["documents"]["selected"] = {"schema": "subset", "payload": reference["where"]}
    reference["where"] = {"subset": "selected"}
    named = compile_project(Project.model_validate(data))
    assert named.runs == plan.runs and named.requests == plan.requests
    assert named.subsets["default", "evaluate", "subject"].document == "selected"

    measured = SubsetTerms.model_validate({"best": {"by": "loss", "k": 1}})
    reference["where"] = measured.model_dump(exclude_unset=True)
    [problem] = compile_project(Project.model_validate(data)).diagnostics
    assert problem.path == "inputs.subject.where.best"
    reference["where"] = subset_to_rows(measured, ["seed=c"]).model_dump(exclude_unset=True)
    frozen = compile_project(Project.model_validate(data))
    assert not frozen.diagnostics
    [reader] = [run for run in frozen.runs if run.kind == "evaluate"]
    assert reader.inputs["subject"].producer.row == "seed=c"
    assert reader.request_id == evaluation_runs[1].request_id
    reference["where"] = {"rows": []}
    empty = compile_project(Project.model_validate(data))
    assert not empty.diagnostics and all(run.kind == "train" for run in empty.runs)
    assert empty.subsets["default", "evaluate", "subject"].members == ()
    reference["where"] = {"axes": {"seed": ["typo"]}}
    [problem] = compile_project(Project.model_validate(data)).diagnostics
    assert problem.path == "inputs.subject.where.axes.seed"
    assert "typo" in problem.detail


def test_sweep_conversions_preserve_requests_and_editing_keeps_run_identity():
    data, _ = _project_data()
    data["steps"]["fit"]["sweep"] = {
        "axes": {
            "seed": {
                "path": "seed",
                "sampler": {"distribution": "test", "seed": 5, "count": 2},
                "draws": {"first": 3, "second": 4},
            },
        }
    }
    data["steps"]["evaluate"]["sweep"] = {"axes": {"subject": {"input": "subject"}}}
    project = Project.model_validate(data)
    before = compile_project(project)
    assert not before.diagnostics
    sweep = project.steps["fit"].sweep
    listed = axis_to_list(sweep.axes["seed"])
    project.steps["fit"].sweep = sweep_to_rows(sweep.model_copy(update={"axes": {"seed": listed}}))
    converted = compile_project(project)
    assert converted == before
    listed.values["second"].patches["seed"].value = 9
    listed.values["second"].value = 9
    edited = compile_project(project)
    assert not edited.diagnostics
    assert {run.key for run in edited.runs} == {run.key for run in before.runs}
    changed = {
        run.key
        for run, previous in zip(edited.runs, before.runs, strict=True)
        if run.request_id != previous.request_id
    }
    assert changed == {
        run.key for run in edited.runs if run.coordinates["seed"].value_id == "second"
    }


def test_exception_changes_only_selected_run_requests_and_exclusion_removes_run_key():
    data, _ = _project_data()
    del data["steps"]["evaluate"]
    fit = data["steps"]["fit"]
    fit["sweep"] = {"axes": {"seed": {"path": "seed", "values": [1, 2, 3]}}}
    base = compile_project(Project.model_validate(data))
    assert not base.diagnostics
    first, changed, excluded = base.runs
    fit["sweep"]["exceptions"] = {
        "replace_policy": {
            "where": {"rows": [changed.key.row]},
            "patches": {"on_nan": {"value": "halt_restore_checkpoint"}},
        },
        "omit": {"where": {"rows": [excluded.key.row]}, "exclude": True},
    }
    amended = compile_project(Project.model_validate(data))
    assert not amended.diagnostics
    assert len(amended.runs) == 2
    assert amended.runs[0] == first
    assert amended.runs[1].key == changed.key
    assert amended.runs[1].coordinates == changed.coordinates
    assert amended.runs[1].request_id != changed.request_id


def test_source_projection_uses_last_writer_and_only_the_requested_run_keys():
    data, _ = _project_data()
    fit = data["steps"]["fit"]
    fit["deltas"] = [
        {
            "layer_id": "settings",
            "patches": [{"path": "on_nan", "value": "halt_restore_checkpoint"}],
        }
    ]
    fit["sweep"] = {
        "axes": {
            "seed": {
                "values": {
                    "a": {"patches": {"seed": {"value": 1}}},
                    "b": {"patches": {"seed": {"value": 2}}},
                }
            }
        },
        "exceptions": {
            "retry": {
                "where": {"rows": ["seed=a"]},
                "patches": {"on_nan": {"value": "halt_restore_checkpoint"}},
            }
        },
    }
    data["steps"]["child"] = {"kind": "train", "base": {"step": "fit"}}
    data["steps"]["evaluate"]["inputs"]["subject"].update({"step": "child", "row": "seed=a"})
    requested = RunKey("default", "child", "seed=a")
    consumer = RunKey("default", "evaluate", "default")
    project = Project.model_validate(data)
    plan = compile_project(project, source_runs=[requested, consumer])
    assert not plan.diagnostics
    assert set(plan.sources) == {requested, consumer}
    sources = plan.sources[requested]
    assert (
        sources["/payload/graph/schema_id"].kind,
        sources["/payload/graph/schema_id"].name,
        sources["/payload/graph/schema_id"].path,
    ) == ("document", "model", "/schema_id")
    seed = sources["/payload/seed"]
    assert (seed.kind, seed.step, seed.axis, seed.value) == ("sweep_value", "fit", "seed", "a")
    policy = sources["/payload/on_nan"]
    assert (policy.kind, policy.step, policy.name) == ("exception", "fit", "retry")
    bound = plan.sources[consumer]["/inputs/subject"]
    assert (bound.kind, bound.variant, bound.step, bound.row) == (
        "upstream",
        "default",
        "child",
        "seed=a",
    )
    without_sources = compile_project(project)
    assert plan.runs == without_sources.runs and plan.requests == without_sources.requests

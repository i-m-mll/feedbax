import json
from pathlib import Path

from feedbax.contracts.project import Project
from feedbax.contracts.project_io import (
    ProjectConfiguration,
    StudioProjectFile,
    dump_studio_project,
    dump_text_project,
    load_text_project,
    parse_studio_project,
)
from feedbax.contracts.project_plan import RunKey, compile_project
from feedbax.contracts.project_resolution import ResolutionCode
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
    assert train.key == RunKey("fit", "default")
    assert train.request_id == expected_train
    assert evaluation.key == RunKey("evaluate", "default")
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


def test_invalid_row_is_diagnosed_without_rebinding_the_consumer():
    data, _ = _project_data()
    data["steps"]["fit"]["sweep"] = {
        "rows": {
            "good": {},
            "bad": {"patches": {"seed": {"value": "text"}}},
        }
    }
    plan = compile_project(Project.model_validate(data))
    assert [run.key for run in plan.runs] == [RunKey("fit", "good")]
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

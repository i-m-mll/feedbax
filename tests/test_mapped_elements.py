"""A mapped operation is one element with several requests: member ids by label name, addresses, projections."""

from copy import deepcopy

from feedbax.cli import main
from feedbax.contracts.project import Project
from feedbax.host import HostClient
from feedbax.web.services.project_folder import ProjectFolderService
from feedbax.web.services.project_plan import ProjectPlanRead
from tests.test_canonical_evaluate_admission import _save_project
from tests.test_list_inputs import _two_model_definition
from tests.test_map_dims import MASSES, _card, _compile, _install_probe, _operations_of, _plan

PROJECT = "mapped"


def _mapped(operation="map_probe.align_each", **params):
    definition = _two_model_definition(MASSES)
    align = _operations_of(definition)["align"]
    align["operation"] = operation
    align["params"] |= params
    return definition


def _members(definition):
    return {element.key.member: element.request_id for element in _card(_plan(definition)).requests("align")}


def test_member_ids_name_label_values_and_a_model_edit_changes_only_the_requests_that_read_it():
    definition = _mapped()
    before = _members(definition)
    assert len(before) == 8
    assert "source=conditions,model=train,mass=nominal" in before
    edited = deepcopy(definition)
    edited["steps"]["train"]["base"]["inline"]["training_config"]["n_batches"] += 1
    after = _members(edited)
    assert after.keys() == before.keys()
    changed = {member for member in before if before[member] != after[member]}
    assert changed == {member for member in before if ",model=train," in member}
    assert len(changed) == 4


def test_whole_taken_roles_with_different_labels_carry_the_role_the_output_names_or_are_refused():
    definition = _mapped()
    operations = _operations_of(definition)
    whole = _operations_of(_two_model_definition(MASSES))["align"]["operation"]
    operations["pair"] = {
        "operation": "map_probe.pair", "params": {"ddof": 0, "bounds_scale": 1.0},
        "inputs": {"left": {"input": "states"}, "right": {"element": "align", "product": "velocity"}},
    }
    (problem,) = (problem for problem in _compile(definition).diagnostics if problem.element == "pair")
    assert problem.path == "payload.operations.pair"
    operations["pair"]["operation"] = "map_probe.pair_left"
    card = _card(_plan(definition))
    (paired,) = card.requests("pair")
    assert {entry.key for entry in paired.entries} == {binding.member.key for binding in paired.inputs["left"]}
    assert len(paired.entries) == 8
    operations["pair"]["operation"] = "map_probe.pair"
    operations["align"]["operation"] = whole
    assert not [problem for problem in _compile(definition).diagnostics if problem.element == "pair"]


def _status(capsys, root, *addresses):
    assert main(["status", str(root), *addresses]) == 0, capsys.readouterr()
    return [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()[1:]]


def test_a_mapped_element_is_one_entry_in_each_projection_and_addresses_select_its_requests(
    tmp_path, monkeypatch, capsys,
):
    operations = _install_probe(tmp_path, monkeypatch)
    monkeypatch.setattr(
        HostClient, "jobs_snapshot", lambda self, **_: {"cursor": 0, "oldest_cursor": 0, "jobs": []},
    )
    monkeypatch.setenv("FEEDBAX_HOST_ROOT", str(tmp_path / "host"))
    folders = ProjectFolderService(tmp_path / "projects", operations=operations)
    root = tmp_path / "projects" / PROJECT
    _save_project(folders, PROJECT, Project.model_validate(_mapped()))

    view = ProjectPlanRead.without_history(folders.read_plan(PROJECT, revision=None))
    (row,) = (row for row in view.status_rows() if getattr(row.key, "name", None) == "align")
    assert len(row.requests) == 8
    (element,) = (element for element in view.window_map().analysis_elements if element.name == "align")
    assert len(element.requests) == 8
    assert element.requests[0].labels == {"source": "conditions", "model": "train", "mass": "nominal"}

    assert len(printed := _status(capsys, root, "align")) == 8
    assert "analyze/align:source=conditions,model=train,mass=nominal" in printed
    assert _status(capsys, root, printed[0]) == [printed[0]]
    assert len(_status(capsys, root, "align:model=train")) == 4
    assert main(["status", str(root), "align:model=nobody"]) == 2
    assert "names nothing" in capsys.readouterr().err

    _save_project(folders, PROJECT, Project.model_validate(_mapped("map_probe.align_over", whole=["source", "mass"])))
    assert _status(capsys, root, "align:model=train") == ["analyze/align:model=train"]
    _save_project(folders, PROJECT, Project.model_validate(_mapped("map_probe.align_over", whole=["source"])))
    assert sorted(_status(capsys, root, "align:model=train")) == [
        f"analyze/align:model=train,mass={mass}" for mass in ("heavier", "heavy", "light", "nominal")
    ]

    for row in _status(capsys, root, "conditions"):
        assert "==" not in row and "@" not in row.partition(":")[2]
        assert _status(capsys, root, row) == [row]

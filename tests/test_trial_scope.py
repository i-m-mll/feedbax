"""An edit for a selection of runs is stored on the exception whose selector picks them."""

import pytest

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.studio_project import AuthoredFileAddress
from feedbax.web.services.project_folder import ProjectFolderInvalid
from feedbax.web.services.trial_scope import row_exception_file, values_exception_file

_ADDRESS = AuthoredFileAddress(kind="step", id="train")
_MASS = "trial_documents.model.mass"


def _edit(raw, rows, value):
    (patch,) = row_exception_file(_ADDRESS, raw, rows, [OverridePatch(path=_MASS, op="replace", value=value)]).patches
    return raw | {"sweep": patch.value}


def test_an_edit_finds_its_exception_by_the_runs_it_picks():
    raw = {"sweep": {"axes": {"seed": {"values": {"one": {}, "two": {}}}}}}
    raw = _edit(raw, ["seed=two", "seed=one"], 3)
    assert raw["sweep"]["exceptions"] == {"exception": {
        "where": {"rows": ["seed=one", "seed=two"]}, "patches": {_MASS: {"op": "replace", "value": 3}},
    }}
    raw = _edit(raw, ["seed=one", "seed=two"], 4)
    assert list(raw["sweep"]["exceptions"]) == ["exception"]
    assert raw["sweep"]["exceptions"]["exception"]["patches"][_MASS]["value"] == 4
    raw = _edit(raw, ["seed=one"], 5)
    assert list(raw["sweep"]["exceptions"]) == ["exception", "exception-2"]


def test_an_authored_exception_over_the_same_runs_is_changed_under_its_own_name():
    raw = {"sweep": {"exceptions": {"mine": {"where": {"rows": ["seed=two", "seed=one"]}}}}}
    raw = _edit(raw, ["seed=one", "seed=two"], 3)
    assert list(raw["sweep"]["exceptions"]) == ["mine"]
    assert raw["sweep"]["exceptions"]["mine"]["patches"] == {_MASS: {"op": "replace", "value": 3}}

    values = {"sweep": {"exceptions": {"slow": {"where": {"axes": {"rate": ["slow", "fast"], "seed": ["one"]}}}}}}
    (patch,) = values_exception_file(_ADDRESS, values, {"seed": ["one"], "rate": ["fast", "slow"]}, [
        OverridePatch(path=_MASS, op="replace", value=3),
    ]).patches
    assert list(patch.value["exceptions"]) == ["slow"]


def test_an_exclusion_over_the_runs_is_not_edited():
    raw = {"sweep": {"exceptions": {"gone": {"where": {"rows": ["seed=one"]}, "exclude": True}}}}
    with pytest.raises(ProjectFolderInvalid):
        _edit(raw, ["seed=one"], 3)

from copy import deepcopy

import pytest
from hypothesis import given, strategies as st
from pydantic import ValidationError

from feedbax.contracts.manifest import OverridePatch
from feedbax.contracts.matrix_core import (
    apply_override_patch_in_place,
    apply_stored_patch_copy,
    apply_stored_patch_in_place,
)
from feedbax.contracts.project import (
    BasedDocument,
    DifferenceNotApplied,
    OrderDocumentMembers,
    RemoveDocumentValue,
    SetDocumentValue,
    SetProjectValue,
    apply_document_edit,
    apply_differences,
    diff,
)
from feedbax.contracts.set_document import SetEntry


def _keyed_collection(parts):
    return parts == ("members",) or len(parts) == 3 and parts[0] == "members" and parts[2] == "children"


_keys = st.sampled_from(["a", "b", "c", "d", "e", "slash/key", "tilde~key"])
_members = st.dictionaries(
    _keys, st.fixed_dictionaries({"children": st.dictionaries(_keys, st.integers(-20, 20), max_size=5)}), max_size=5,
)


@given(_members, _members)
def test_member_differences_reconstruct_nested_keyed_collections(base_members, target_members):
    base, target = {"members": base_members}, {"members": target_members}
    changes = diff(base, target, keyed_collection=_keyed_collection)
    actual = apply_differences(base, changes, keyed_collection=_keyed_collection)

    assert actual == target
    assert list(actual["members"]) == list(target_members)
    for key, member in actual["members"].items():
        assert list(member["children"]) == list(target_members[key]["children"])
    assert diff(target, target, keyed_collection=_keyed_collection) == {}


def test_order_runs_after_edits_and_keeps_new_base_members():
    base = {"members": {"b": 2, "a": 1, "dropped": 3, "future-one": 4, "future-two": 5}}
    authored = BasedDocument.model_validate({
        "schema": "example.members",
        "label": "Reordered",
        "base": {"document": "base"},
        "differences": {
            "/members": {"order": ["new", "a", "no-longer-in-base", "b"]},
            "/members/dropped": {"op": "remove"},
            "/members/new": {"value": 6},
            "/members/b": {"value": 7},
        },
    })
    actual = apply_differences(base, authored.differences, keyed_collection=_keyed_collection)

    assert list(actual["members"].items()) == [
        ("new", 6), ("a", 1), ("b", 7), ("future-one", 4), ("future-two", 5),
    ]
    assert list(base["members"].items()) == [
        ("b", 2), ("a", 1), ("dropped", 3), ("future-one", 4), ("future-two", 5),
    ]


def test_diff_emits_member_edits_before_the_collection_order():
    base = {"members": {"a": 1, "b": 2, "drop": 3}, "metadata": {"first": 1, "second": 2}}
    target = {"members": {"b": 4, "add": 5, "a": 1}, "metadata": {"second": 2, "first": 1}}
    changes = diff(base, target, keyed_collection=_keyed_collection)

    assert list(changes.items()) == [
        ("/members/b", SetDocumentValue(value=4)),
        ("/members/add", SetDocumentValue(value=5)),
        ("/members/drop", RemoveDocumentValue(op="remove")),
        ("/members", OrderDocumentMembers(order=["b", "add", "a"])),
    ]


def test_editing_one_member_stays_local_as_the_collection_grows():
    for count in (10, 1_000):
        base = {"members": {f"member-{index}": {"value": index} for index in range(count)}}
        target = deepcopy(base)
        target["members"]["member-2"]["value"] = 57
        changes = diff(base, target, keyed_collection=_keyed_collection)
        assert changes == {"/members/member-2/value": SetDocumentValue(value=57)}

        base["members"]["member-1"]["value"] = 23
        base["members"]["future"] = {"value": 40}
        actual = apply_differences(base, changes, keyed_collection=_keyed_collection)
        assert actual["members"]["member-1"] == {"value": 23}
        assert actual["members"]["member-2"] == {"value": 57}
        assert actual["members"]["future"] == {"value": 40}


def test_order_requires_a_typed_collection_and_cannot_look_inside_replaced_value():
    with pytest.raises(DifferenceNotApplied, match="no keyed collection at /metadata"):
        apply_differences(
            {"metadata": {"a": 1}}, {"/metadata": OrderDocumentMembers(order=["a"])},
            keyed_collection=_keyed_collection,
        )
    with pytest.raises(DifferenceNotApplied, match="no object at /members"):
        apply_differences({}, {"/members": OrderDocumentMembers(order=[])}, keyed_collection=_keyed_collection)

    with pytest.raises(ValidationError, match="lies under"):
        BasedDocument.model_validate({
            "schema": "example.members",
            "label": "Replaced",
            "base": {"document": "base"},
            "differences": {
                "/members": {"order": ["a"]},
                "/members/a": {"value": {"children": {}}},
                "/members/a/children": {"order": []},
            },
        })


def test_remove_of_an_absent_leaf_still_requires_its_parent():
    payload = {"member": {"kept": 1}}
    absent = RemoveDocumentValue(op="remove")
    assert apply_differences(payload, {"/member/absent": absent}) == payload
    with pytest.raises(DifferenceNotApplied, match="no object at /missing"):
        apply_differences(payload, {"/missing/absent": absent})

    absent_leaf = OverridePatch(op="remove", path="member.absent")
    missing_parent = OverridePatch(op="remove", path="missing.absent")
    assert apply_stored_patch_copy(payload, absent_leaf) == payload
    for apply_patch in (apply_stored_patch_in_place, apply_stored_patch_copy):
        apply_patch(payload, absent_leaf)
        with pytest.raises(ValueError, match="cannot traverse missing segment"):
            apply_patch(payload, missing_parent)
    with pytest.raises(ValueError, match="targets a missing key"):
        apply_override_patch_in_place(payload, absent_leaf)


def test_stored_sweeps_write_keyed_members_while_lowered_variation_can_use_ids():
    patch = OverridePatch(op="replace", path="members.first.value", value=2)
    stored = {"members": {"first": {"value": 1}}}
    assert apply_stored_patch_copy(stored, patch) == {"members": {"first": {"value": 2}}}
    assert stored == {"members": {"first": {"value": 1}}}

    lowered = {"members": [{"id": "first", "value": 1}]}
    with pytest.raises(ValueError, match="cannot traverse an array"):
        apply_stored_patch_in_place(lowered, patch)
    apply_override_patch_in_place(lowered, patch)
    assert lowered == {"members": [{"id": "first", "value": 2}]}


def test_only_expression_arguments_accept_stored_array_indices():
    payload = {
        "variable": {"kind": "call", "function": "vector", "args": [{"kind": "literal", "value": 1}]},
        "weights": [1],
    }
    expression_write = OverridePatch(path="variable.args.0.value", value=2)
    apply_stored_patch_in_place(payload, expression_write)
    assert payload["variable"]["args"][0]["value"] == 2

    with pytest.raises(ValueError, match="cannot traverse an array"):
        apply_stored_patch_copy(payload, OverridePatch(path="weights.0", value=2))
    with pytest.raises(ValueError, match="non-negative decimal index"):
        apply_stored_patch_copy(payload, OverridePatch(path="variable.args.00.value", value=2))

    apply_document_edit(payload, SetProjectValue(op="add", path="/weights/0", value=2), 2)
    assert payload["weights"] == [2, 1]


def test_link_members_require_pointers_and_link_named_variables_are_addressable():
    payload = {
        "links": {"inputs.load": "old-variable"},
        "task_links": {},
        "variables": {"links": {"label": "link values", "type": {"base": "real"}, "value": 1}},
    }
    with pytest.raises(ValueError, match="Link members require a JSON pointer"):
        apply_stored_patch_copy(payload, OverridePatch(path="links.inputs.load", value="new-variable"))
    with pytest.raises(ValueError, match="Link members require a JSON pointer"):
        apply_stored_patch_in_place(payload, OverridePatch(op="remove", path="task_links.old"))

    derived = apply_differences(payload, {"/links/inputs.load": SetDocumentValue(value="new-variable")})
    assert derived["links"] == {"inputs.load": "new-variable"}
    apply_stored_patch_in_place(payload, OverridePatch(path="variables.links.value", value=2))
    assert payload["variables"]["links"]["value"] == 2


def test_a_set_entry_cannot_store_a_collection_order():
    with pytest.raises(ValidationError):
        SetEntry.model_validate({"label": "Entry", "differences": {"/members": {"order": ["member"]}}})

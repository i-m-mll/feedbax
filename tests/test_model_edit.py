"""Addressed operations and the model-document transaction invariants."""

from hypothesis import settings, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule
from fastapi.testclient import TestClient
import pytest

from feedbax.contracts.canonical_json import canonical_json_v2_bytes
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.model_document import SignalType
from feedbax.contracts.model_edit import ModelEditCommand, ModelEditProblem, ModelEditState
from feedbax.contracts.project import Document, Project
from feedbax.contracts.project_io import ProjectConfiguration, dump_text_project
from feedbax.studio.model_edit import Port, apply_model_edit
from feedbax.web.app import create_app

SCALAR = Port(SignalType(base="scalar", shape=()))


def _address(scope=(), kind="graph", **fields):
    return {"document": "model", "composites": scope, "element": {"kind": kind, **fields}}


def _end(component, port):
    return {"owner": "component", "component": component, "port": port}


def _leaf(type_="leaf", **fields):
    return {"kind": "leaf", "name": "Same", "type": type_, "fields": fields}


def _ports(component, direction, port):
    if component["type"] == "mass":
        return Port("translational") if direction == "physical" and port == "flange" else None
    return SCALAR if (direction, port) in {("input", "in"), ("output", "out")} else None


def _accepts(component, domain):
    return component["kind"] == "composite" or (component["type"] == "mass") == (domain is not None)


def _execute(state, op, **fields):
    command = ModelEditCommand.model_validate({"operation": {"op": op, **fields}})
    return apply_model_edit("model", state, command, ports=_ports, accepts=_accepts)


def _bytes(state):
    return canonical_json_v2_bytes(state.model_dump(mode="json"))


def _start(deep_name="inner", *, decorated=False, fanout=False, default=False):
    delay = {"kind": "delay", "id": "delay", "delay": {"value": 1, "unit": "st"}}
    inner = {"components": {"same": _leaf("leaf", gain=1), "sink": _leaf()}}
    comp = {"kind": "composite", "name": deep_name, "ports": {
        "inputs": [{"id": "in", "name": "In", "type": {}, **({"default": {"value": 0}} if default else {})}],
        "outputs": [{"id": "out", "name": "Out", "type": {}}],
    }, "interior": inner}
    inner["wires"] = {
        "input": {"source": {"owner": "inputs", "port": "in"}, "target": _end("same", "in"),
                  "attachments": [{"kind": "probe", "id": "inner-probe", "name": "Read"}]},
        "output": {"source": _end("same", "out"), "target": {"owner": "outputs", "port": "out"},
                   "attachments": [{"kind": "probe", "id": "output-probe", "name": "Write"}]},
    }
    if fanout:
        inner["wires"]["fanout"] = {"source": {"owner": "inputs", "port": "in"}, "target": _end("sink", "in")}
    components = {"same": _leaf(), "sink": _leaf(), "child": comp}
    wires = {
        "out": {"source": _end("child", "out"), "target": _end("sink", "in"),
                "attachments": [{"kind": "intervention", "id": "outer-transform", "operation": "Scale"}]},
    }
    if not default:
        wires["in"] = {"source": _end("same", "out"), "target": _end("child", "in"),
                       "attachments": [delay] if decorated else []}
    outer = {"kind": "composite", "name": "Outer", "ports": {},
             "interior": {"components": components, "wires": wires}}
    physics = {"kind": "composite", "name": "Physics",
               "ports": {"physical": [{"id": "frame", "name": "Frame", "type": "translational"}]},
               "interior": {"physical_domain": "translational", "components": {"mass": _leaf("mass")}}}
    return ModelEditState.model_validate({
        "document": {"step_seconds": 0.01, "ports": {}, "interior": {
            "components": {"outer": outer, "same": _leaf(), "sink": _leaf(), "physics": physics},
        }},
        "positions": {
            ModelAddress.model_validate(_address(scope, "component", id="same")).to_text(): {"x": index, "y": 0}
            for index, scope in enumerate(((), ("outer",), ("outer", "child")))
        },
    })


def test_ungroup_preserves_attachment_order_and_refuses_shared_delay():
    address = _address(("outer",), "component", id="child")
    accepted = _execute(_start(decorated=True), "ungroup", address=address)
    assert not isinstance(accepted, ModelEditProblem), accepted
    wires = accepted.state.document.interior.components["outer"].interior.wires.values()
    assert {tuple(a.id for a in wire.attachments) for wire in wires} == {
        ("delay", "inner-probe"), ("outer-transform", "output-probe"),
    }

    start = _start(decorated=True, fanout=True)
    before = _bytes(start)
    refused = _execute(start, "ungroup", address=address)
    assert isinstance(refused, ModelEditProblem)
    assert refused.address == ModelAddress.model_validate(_address(("outer",), "wire", id="in"))
    assert _bytes(start) == before


@pytest.mark.parametrize("outside", [False, True])
def test_ungroup_refuses_port_attachments(outside):
    state = _start()
    child = state.document.interior.components["outer"].interior.components["child"]
    from feedbax.contracts.model_document import ProbeAttachment
    placement = child.attachments if outside else child.interior.attachments
    placement.sources["out" if outside else "in"] = (
        ProbeAttachment(kind="probe", id="probe", name="Read"),
    )
    before = _bytes(state)
    result = _execute(state, "ungroup", address=_address(("outer",), "component", id="child"))
    assert isinstance(result, ModelEditProblem)
    assert result.address.element.kind == "port"
    assert _bytes(state) == before


def test_ungroup_refuses_an_unconnected_input_default():
    state = _start(default=True)
    result = _execute(state, "ungroup", address=_address(("outer",), "component", id="child"))
    assert isinstance(result, ModelEditProblem)
    assert result.address.element.kind == "port"


def test_addressed_route_settles_document_and_position_and_rejects_raw_patch(tmp_path):
    path = tmp_path / "project"
    project = Project(documents={
        "model": Document(schema="feedbax.spec.model", payload={
            "step_seconds": 0.01, "ports": {},
            "interior": {"components": {"outer": {
                "kind": "composite", "name": "Outer", "ports": {}, "interior": {},
            }}},
        }),
    })
    for name, data in dump_text_project(project, ProjectConfiguration(project="project", store="store")).items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    with TestClient(create_app(projects_dir=tmp_path)) as client:
        url = "/api/projects/project"
        before = client.get(url).json()
        patch = client.patch(url, json={
            "revision": before["revision"],
            "address": {"kind": "document", "id": "model"},
            "patches": [{"op": "replace", "path": "/payload/step_seconds", "value": 0.02}],
        })
        assert patch.status_code == 422, patch.text
        body = {
            "revision": before["revision"], "presentation_revision": before["presentation_revision"],
            "document": "model",
            "command": {"operation": {
                "op": "add_component", "graph": _address(("outer",)),
                "component": {"kind": "leaf", "name": "Scale", "type": "feedbax.Scale"},
                "position": {"x": 2, "y": 3},
            }},
        }
        response = client.post(url + "/model/edit", json=body)
        assert response.status_code == 200, response.text
        result = response.json()["result"]
        assert result["status"] == "applied", result
        address = ModelAddress.model_validate(result["changed"][0])
        reopened = client.get(url + "/open").json()["snapshot"]
        payload = reopened["project"]["documents"]["model"]["payload"]
        outer = payload["interior"]["components"]["outer"]
        assert list(outer["interior"]["components"]) == [address.element.id]
        assert reopened["presentation"]["documents"]["model"]["positions"][address.to_text()] == {
            "x": 2, "y": 3,
        }
        assert client.post(url + "/model/edit", json=body).status_code == 409


def _graphs(owner, scope=()):
    graph = owner.interior
    yield scope, owner, graph
    for id_, component in graph.components.items():
        if component.kind == "composite":
            yield from _graphs(component, (*scope, id_))


class ModelEdits(RuleBasedStateMachine):
    @initialize(inner=st.text(alphabet="abc", min_size=1, max_size=5), decorated=st.booleans())
    def initialize_project(self, inner, decorated):
        self.current = self.initial = _start(inner, decorated=decorated)
        self.undo = []
        self.redo = []
        self.clipboard = None

    def edit(self, operation, **fields):
        before = _bytes(self.current)
        result = _execute(self.current, operation, **fields)
        assert _bytes(self.current) == before
        if isinstance(result, ModelEditProblem):
            assert result.address.document == "model"
            return result
        if result.clipboard is not None:
            self.clipboard = result.clipboard
        if _bytes(result.state) != before:
            self.undo.append(self.current)
            self.redo.clear()
            self.current = result.state
        return result

    def choose(self, data):
        return data.draw(st.sampled_from(list(_graphs(self.current.document))))

    @rule(data=st.data())
    def add(self, data):
        scope, _, graph = self.choose(data)
        self.edit("add_component", graph=_address(scope), component=_leaf(
            "mass" if graph.physical_domain else "leaf", gain=1,
        ), position={"x": 0, "y": 2})

    @rule(data=st.data())
    def selection(self, data):
        scope, _, graph = self.choose(data)
        if not graph.components:
            return
        selection = [_address(scope, "component", id=id_) for id_ in data.draw(
            st.lists(st.sampled_from(tuple(graph.components)), min_size=1, max_size=3, unique=True)
        )]
        action = data.draw(st.sampled_from(["copy", "cut", "delete", "duplicate", "group"]))
        before = self.current
        result = self.edit(action, selection=selection, **({"name": "Group"} if action == "group" else {}))
        if action == "group" and not isinstance(result, ModelEditProblem):
            address = result.changed[-1]
            restored = self.edit("ungroup", address=address)
            assert not isinstance(restored, ModelEditProblem), restored
            assert _bytes(self.current) == _bytes(before)

    @rule(data=st.data())
    def paste(self, data):
        scope, _, _ = self.choose(data)
        if self.clipboard is not None:
            self.edit("paste", graph=_address(scope), clipboard=self.clipboard)

    @rule(data=st.data())
    def wire(self, data):
        scope, _, graph = self.choose(data)
        leaves = [id_ for id_, component in graph.components.items() if component.kind == "leaf"]
        if not leaves:
            return
        a, b = data.draw(st.tuples(st.sampled_from(leaves), st.sampled_from(leaves)))
        if graph.physical_domain:
            self.edit("connect_physical", graph=_address(scope), connection={
                "a": _end(a, "flange"), "b": {"owner": "boundary", "port": "frame"},
            })
        else:
            self.edit("connect", graph=_address(scope), source=_end(a, "out"), target=_end(b, "in"))

    @rule(data=st.data())
    def bring_in(self, data):
        scope, _, graph = self.choose(data)
        composites = [id_ for id_, item in graph.components.items() if item.kind == "composite"]
        leaves = [id_ for id_, item in graph.components.items() if item.kind == "leaf"]
        if composites and leaves and graph.physical_domain is None:
            a = data.draw(st.sampled_from(leaves))
            b = data.draw(st.sampled_from(composites))
            self.edit("bring_signal_in", source_graph=_address(scope), source=_end(a, "out"),
                      destination=_address((*scope, b)))

    @rule(data=st.data())
    def port(self, data):
        scope, owner, _ = self.choose(data)
        direction = data.draw(st.sampled_from(["inputs", "outputs"]))
        ports = getattr(owner.ports, direction)
        if not ports or data.draw(st.booleans()):
            self.edit("add_port", graph=_address(scope), direction=direction[:-1], port={
                "id": data.draw(st.sampled_from(["edit-a", "edit-b", "edit-c"])),
                "name": data.draw(st.sampled_from(["A", "B", "C"])), "type": {},
            })
        else:
            port = data.draw(st.sampled_from(ports))
            address = _address(scope, "port", end={"owner": direction, "port": port.id})
            action = data.draw(st.sampled_from(["remove_port", "rename", "move_port"]))
            kwargs = {"name": "Renamed"} if action == "rename" else {"index": 0} if action == "move_port" else {}
            self.edit(action, address=address, **kwargs)

    @rule(data=st.data(), value=st.integers(-3, 5))
    def fields(self, data, value):
        scope, _, graph = self.choose(data)
        if graph.components:
            id_ = data.draw(st.sampled_from(tuple(graph.components)))
            self.edit("set_field", address=_address(scope, "field", parent={"kind": "component", "id": id_},
                                                      path=["fields", "gain"]), value=value)
            self.edit("rename", address=_address(scope, "component", id=id_), name=f"Name {value}")

    @rule()
    def undo_edit(self):
        if self.undo:
            self.redo.append(self.current)
            self.current = self.undo.pop()

    @rule()
    def redo_edit(self):
        if self.redo:
            self.undo.append(self.current)
            self.current = self.redo.pop()

    @invariant()
    def local_ids_and_wire_directions(self):
        for _, owner, graph in _graphs(self.current.document):
            ports = (*owner.ports.inputs, *owner.ports.outputs, *owner.ports.physical)
            assert len(ports) == len({port.id for port in ports})
            connections = getattr(graph, "connections", {})
            assert len(graph.components) + len(graph.wires) + len(connections) == len(
                graph.components.keys() | graph.wires.keys() | connections.keys()
            )
            targets = []
            for wire in graph.wires.values():
                for end, boundary, direction in ((wire.source, "inputs", "output"), (wire.target, "outputs", "input")):
                    if end.owner == "component":
                        component = graph.components[end.component]
                        if component.kind == "composite":
                            assert end.port in {p.id for p in getattr(component.ports, direction + "s")}
                        else:
                            assert _ports(component.model_dump(), direction, end.port) is not None
                    else:
                        assert end.owner == boundary
                        assert end.port in {p.id for p in getattr(owner.ports, boundary)}
                targets.append(wire.target.model_dump_json())
            assert len(targets) == len(set(targets))
            for connection in connections.values():
                for end in (connection.a, connection.b):
                    if end.owner == "component":
                        assert end.component in graph.components
                    else:
                        assert end.port in {p.id for p in owner.ports.physical}

    def teardown(self):
        final = _bytes(self.current)
        steps = len(self.undo)
        for _ in range(steps):
            self.undo_edit()
        assert _bytes(self.current) == _bytes(self.initial)
        for _ in range(steps):
            self.redo_edit()
        assert _bytes(self.current) == final


TestModelEdits = ModelEdits.TestCase
TestModelEdits.settings = settings(max_examples=30, stateful_step_count=22, deadline=None)

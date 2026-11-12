"""The Studio walkthrough's small CPU reaching project."""

from __future__ import annotations

import argparse

from feedbax.compiler.normalization import stamp_component_versions
from feedbax.compiler.templates import network_template_graph
from feedbax.component_registry import ComponentRegistry
from feedbax.contracts.graph import (
    STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_ID,
    STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_VERSION,
    ComponentSpec,
    GraphMetadata,
    GraphUIState,
    SemanticAnchor,
    StudioPersistenceDocument,
    StudioTaskBinding,
    StudioTaskBindingSpec,
    StudioTaskDataSpec,
    StudioValueSpec,
    WireSpec,
    WorkspaceDocument,
    build_default_studio_workspace,
    studio_semantic_document_sha256,
)

from feedbax.contracts.request_specs import TrainingRequestSpec
from feedbax.contracts.studio_training import STUDIO_GRAPH_METHOD
from feedbax.contracts.training import TaskSpec, TrainingSpec
from feedbax.studio.protocol import infer_task_n_steps
from feedbax.studio.training_requests import studio_graph_training_settings


NAME = "Point-mass reach"
STEPS = 100
BATCHES = 30


def reaching_project(*, batches: int = BATCHES, name: str = NAME) -> StudioPersistenceDocument:
    graph = network_template_graph({"input_size": 2, "hidden_size": 100, "out_size": 2})
    del graph.nodes["input_mux"]
    graph.wires = [wire for wire in graph.wires if wire.source_node != "input_mux"]
    graph.input_ports = []
    graph.input_bindings = {}
    graph.nodes["plant"] = ComponentSpec(
        type="feedbax.PointMass",
        params={"mass": 1.0, "dt": 0.01},
        input_ports=["force"],
        output_ports=["effector", "state"],
    )
    graph.wires.append(
        WireSpec(
            source_node="readout", source_port="output", target_node="plant", target_port="force"
        )
    )
    graph.output_ports = ["effector"]
    graph.output_bindings = {"effector": ("plant", "effector")}
    graph.metadata = GraphMetadata(
        name=name,
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )
    graph = stamp_component_versions(graph, ComponentRegistry())

    workspace = build_default_studio_workspace(label=name)
    scenario = workspace.scenarios["scenario:train"]
    scenario.task_spec = {"type": "SimpleReaches", "params": {"n_steps": STEPS}}
    scenario.task_binding_spec = StudioTaskBindingSpec(
        exposed_data=[
            StudioTaskDataSpec(
                id="reach_target",
                label="Reach target",
                kind="signal",
                role="model_input",
                path="inputs.effector_target.pos",
                bindable=True,
                expected_shape=["time", 2],
                value_spec=StudioValueSpec(
                    value_form="literal", mode="constant", value=[0.1, -0.05], dtype="float32",
                ),
            ),
        ],
        bindings=[
            StudioTaskBinding(
                id="task:reach_target->cell:input",
                source_data_id="reach_target",
                target_node_id="cell",
                target_port="input",
                role="model_input",
            ),
        ],
    )
    scenario.training_spec = {
        "optimizer": {"type": "adam", "params": {"learning_rate": 0.001}},
        "loss": {
            "type": "TargetStateLoss",
            "label": "Endpoint position",
            "selector": "path:plant.effector.pos",
            "target_selector": "task_data:reach_target",
            "weight": 1.0,
            "norm": "squared_l2",
            "time_agg": {"mode": "final"},
        },
        "n_batches": batches,
        "batch_size": 1,
    }
    positions = {"cell": (70, 100), "readout": (250, 100), "plant": (430, 100)}
    ui_state = GraphUIState.model_validate({
        "node_states": {
            node_id: {"position": {"x": x, "y": y}} for node_id, (x, y) in positions.items()
        },
    })
    return StudioPersistenceDocument(
        schema_id=STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_ID,
        schema_version=STUDIO_PERSISTENCE_DOCUMENT_SCHEMA_VERSION,
        graph=graph,
        workspace=workspace,
        workspace_document=WorkspaceDocument(
            semantic_root=SemanticAnchor(
                semantic_document_sha256=studio_semantic_document_sha256(graph, workspace),
                authored_path="/graph",
            ),
            graph_ui_state=ui_state,
        ),
    )


def reaching_train_request(project: StudioPersistenceDocument) -> TrainingRequestSpec:
    scenario = project.workspace.scenarios["scenario:train"]
    task = TaskSpec.model_validate(scenario.task_spec)
    training = TrainingSpec.model_validate(scenario.training_spec)
    settings = studio_graph_training_settings(training, scenario.training_spec, infer_task_n_steps(task))
    return TrainingRequestSpec(
        graph=project.graph.model_copy(update={"metadata": None}),
        task=task,
        task_binding_spec=scenario.task_binding_spec,
        objective=settings.objective,
        training_config=settings.config,
        method_ref=STUDIO_GRAPH_METHOD,
        method_payload=settings.method_payload,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=BATCHES)
    parser.add_argument("--name", default=NAME)
    args = parser.parse_args()
    print(reaching_project(batches=args.batches, name=args.name).model_dump_json(exclude_none=True))

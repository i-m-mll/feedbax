"""Named document inputs for the managed Train lifecycle witnesses."""

from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.project import InlineBase, Project
from feedbax.contracts.rollout import ROLLOUT_TYPE
from feedbax.examples import load_shipped_projects


COMMAND_ADDRESS = ModelAddress.model_validate({
    "document": "point-mass", "composites": [], "element": {
        "kind": "port", "end": {"owner": "outputs", "port": "command"},
    },
}).to_text()


def managed_train_project(*, batches: int = 2, scenario: str = "simple-reaches") -> Project:
    example = load_shipped_projects()["point-mass-reaching"]
    step = example.steps[scenario]
    assert isinstance(step.base, InlineBase)
    payload = step.base.inline
    payload["training_config"]["n_batches"] = batches
    payload["method_payload"]["payload"]["target_total"] = batches
    trial = payload["trial_documents"]
    names = {trial[field]["document"] for field in ("model", "task", "objective")}
    model = example.documents[trial["model"]["document"]].payload
    interior = model["interior"]
    # Every leaf in this smaller Model has a semantic checkpoint representation.
    for name in ("position", "velocity", "feedback", "flatten"):
        del interior["components"][name]
    interior["components"]["cell"]["fields"]["input_size"] = 2
    interior["wires"] = {
        name: interior["wires"][name] for name in ("target", "decode", "drive", "command")
    }
    interior["wires"]["target"]["target"] = {"owner": "component", "component": "cell", "port": "input"}
    model["ports"]["outputs"] = [
        port for port in model["ports"]["outputs"] if port["id"] == "command"
    ]
    objective = example.documents[trial["objective"]["document"]].payload
    objective["terms"] = [{
        "id": "command",
        "measure": {
            "kind": "call", "function": "sqnorm",
            "args": [{"kind": "name", "reference": COMMAND_ADDRESS}],
        },
        "reduction": "last",
    }]
    return Project(
        documents={name: document for name, document in example.documents.items() if name in names},
        steps={"train": step},
    )


def managed_train_payload(*, batches: int = 2, scenario: str = "simple-reaches") -> dict:
    project = managed_train_project(batches=batches, scenario=scenario).model_dump(
        mode="json", by_alias=True, exclude_unset=True,
    )
    payload = project["steps"]["train"]["base"]["inline"]
    documents = payload["trial_documents"]
    for name in ("model", "task", "objective"):
        documents[name] = project["documents"][documents[name]["document"]]["payload"]
    return payload


def managed_evaluation_step(subject: str) -> dict:
    return {
        "kind": "evaluate",
        "base": {"inline": {
            "evaluation_type": ROLLOUT_TYPE,
            "params": {
                "seed": 731, "trial_policy": "paired", "variation": [],
                "trial_set": {"kind": "every_condition"},
            },
        }},
        "inputs": {"subject": {"step": subject, "checkpoint": "last"}},
    }


def managed_train_spec(*, batches: int = 2):
    from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
    from feedbax.plugins.application import new_application_registry_bundle
    from feedbax.studio.canonical_train import lower_canonical_train

    registries = new_application_registry_bundle()
    spec = lower_canonical_train(
        DocumentTrainingRequestSpec.model_validate(managed_train_payload(batches=batches)),
        registries.training_programs,
        registries=registries,
    )
    return spec, registries

"""A synthesis method fills the shipped graph's declared model parameters."""

import jax
import numpy as np

from feedbax.application.composition import load_application
from feedbax.contracts.model_address import ModelAddress
from feedbax.contracts.request_specs import SynthesisRequestSpec, TrialDocuments
from feedbax.examples import load_shipped_projects
from feedbax.host.synthesis import SynthesisDelivery, synthesize_model
from feedbax.persistence.array_container import decode_array_container
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.model_tree import MODEL_CHECKPOINT_STORE, read_model_tree


def test_synthesis_writes_one_replica_with_every_parameter(tmp_path):
    project = load_shipped_projects()["point-mass-reaching"]
    spec = SynthesisRequestSpec(
        trial_documents=TrialDocuments.model_validate({
            "model": project.documents["point-mass"].payload,
            "task": project.documents["simple-reaches-task"].payload,
            "objective": project.documents["simple-reaches-objective"].payload,
        }),
        method="feedbax.constant_parameters",
        settings={"value": 0.125},
    )
    registry = load_application().bundle
    artifacts = ImmutableArtifactBlobProvider(tmp_path / "artifacts")
    transaction = tmp_path / "attempt"
    with jax.enable_x64():
        result = synthesize_model(spec, registry=registry, artifacts=artifacts, transaction=transaction)
    assert isinstance(result, SynthesisDelivery), result
    store = ImmutableArtifactBlobProvider(transaction / MODEL_CHECKPOINT_STORE)
    model = read_model_tree(store, result.model.tree, result.model.encoding)
    replica, = model.replicas
    assert replica.descriptor.id == "0"
    arrays = decode_array_container(
        store.get_object(replica.arrays), expected_descriptor_sha256=replica.descriptor.array_descriptor_sha256,
    )
    parameters = [
        arrays.tensors[address] for address, metadata in arrays.descriptor.metadata.items()
        if metadata["representation"] == "jax-array"
    ]
    assert parameters
    for array in parameters:
        assert array.dtype == np.float64
        np.testing.assert_equal(array, np.full_like(array, 0.125))

    restricted = spec.model_copy(update={"target": ModelAddress.at_stored_path(("interior", "components", "plant"))})
    with jax.enable_x64():
        problems = synthesize_model(restricted, registry=registry, artifacts=artifacts, transaction=transaction)
    assert isinstance(problems, tuple) and problems
    assert all(problem.model_address is not None and "parameters" in problem.path for problem in problems)

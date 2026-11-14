"""Checks for managed publication of Store content tree views."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from feedbax.contracts.artifact_custody import ArtifactBlobIntegrityError, ContentBinding
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.content_materialization import (
    InputMaterializationError,
    compile_content_input_projection,
    materialize_content_input_projection,
)


def _content_bindings(tmp_path: Path) -> tuple[tuple[ContentBinding, ...], ImmutableArtifactBlobProvider]:
    provider = ImmutableArtifactBlobProvider(tmp_path / "content-store")
    selected = tmp_path / "selected"
    selected.mkdir()
    (selected / "checkpoint.json").write_bytes(b"checkpoint")
    (selected / "manifest.json").write_text("manifest", encoding="utf-8")
    tree = provider.ingest_tree(selected)
    return (
        ContentBinding(binding_name="paired_trial_bank", root_kind="manifest-store", tree=tree),
        ContentBinding(binding_name="training", root_kind="manifest-store", tree=tree),
    ), provider


def test_content_source_mutation_fails_before_materialization(tmp_path: Path) -> None:
    bindings, provider = _content_bindings(tmp_path)
    member = provider.get_tree(bindings[0].tree).files[0].object
    canonical = provider.root / "artifacts" / "sha256" / member.sha256[:2] / member.sha256
    canonical.chmod(0o600)
    canonical.write_bytes(b"x" * member.size_bytes)
    with pytest.raises(ArtifactBlobIntegrityError, match="SHA-256 mismatch"):
        compile_content_input_projection(bindings, provider)


def test_local_content_views_are_isolated_from_each_other_and_custody(tmp_path: Path) -> None:
    bindings, provider = _content_bindings(tmp_path)
    projection = compile_content_input_projection(bindings, provider)

    def materialize(name: str) -> tuple[Path, Path]:
        inputs = tmp_path / name / "inputs"
        views = materialize_content_input_projection(
            projection,
            provider,
            destination_root=inputs / "content" / "trees",
        )
        return inputs, views[0].destination

    first_inputs, first = materialize("first")
    second_inputs, second = materialize("second")
    descriptor_ref = bindings[0].tree.descriptor
    first_descriptor = first_inputs / "content" / "descriptors" / descriptor_ref.sha256
    second_descriptor = second_inputs / "content" / "descriptors" / descriptor_ref.sha256
    original_descriptor = second_descriptor.read_bytes()

    (first / "checkpoint.json").write_bytes(b"experiment")
    first_descriptor.write_bytes(b"changed descriptor")

    assert (second / "checkpoint.json").read_bytes() == b"checkpoint"
    assert second_descriptor.read_bytes() == original_descriptor
    assert provider.get_object(descriptor_ref) == original_descriptor

    _, third = materialize("third")
    assert (third / "checkpoint.json").read_bytes() == b"checkpoint"


@pytest.mark.parametrize("invalid_alias", [0, 1])
def test_content_aliases_reject_conflicting_descriptor_size(
    tmp_path: Path, invalid_alias: int
) -> None:
    bindings, provider = _content_bindings(tmp_path)
    inputs = tmp_path / "view" / "inputs"
    projection = compile_content_input_projection(bindings, provider)
    materialize_content_input_projection(
        projection, provider, destination_root=inputs / "content" / "trees"
    )

    bindings = list(bindings)
    binding = bindings[invalid_alias]
    descriptor = binding.tree.descriptor.model_copy(
        update={"size_bytes": binding.tree.descriptor.size_bytes + 1}
    )
    bindings[invalid_alias] = binding.model_copy(
        update={"tree": binding.tree.model_copy(update={"descriptor": descriptor})}
    )

    with pytest.raises(InputMaterializationError, match="metadata conflicts"):
        compile_content_input_projection(bindings, provider)


def test_concurrent_identical_local_projections_converge(tmp_path: Path) -> None:
    bindings, provider = _content_bindings(tmp_path)
    projection = compile_content_input_projection(bindings, provider)
    destination = tmp_path / "shared-view" / "trees"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _index: materialize_content_input_projection(
                    projection, provider, destination_root=destination
                ),
                range(2),
            )
        )

    assert results[0][0].destination == results[1][0].destination
    assert results[0][0].destination == results[0][1].destination

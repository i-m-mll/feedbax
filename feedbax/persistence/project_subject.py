"""Read pinned Evaluate subjects from the project's declared Store."""

from collections.abc import Callable
from functools import cache
from pathlib import Path

from feedbax.contracts.loaded_model import ImportedModelProvenance, ImportedModelSource, decode_imported_model_document
from feedbax.contracts.model_document import ModelDocument
from feedbax.contracts.project import LoadedModelDocument
from feedbax.contracts.project_resolution import LoadedModelRead, TerminalTrainSource
from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider
from feedbax.persistence.model_tree import ModelTree, read_model_tree


def terminal_train_reader(store_path: Path) -> Callable[[str], TerminalTrainSource | None]:
    @cache
    def provider():
        from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

        return ImmutableArtifactBlobProvider(store_path)

    def read(run: str) -> TerminalTrainSource | None:
        from feedbax.contracts.effective_success import read_success_object, read_success_request
        from feedbax.contracts.request_specs import DocumentTrainingRequestSpec
        from feedbax.persistence.effective_success import terminal_train_subject

        store = provider()
        terminal = terminal_train_subject(store, run)
        if terminal is None:
            return None
        receipt = read_success_object(terminal.receipt, store.get_object)
        decoded = read_success_request(receipt.request, store.get_object)
        if not isinstance(decoded.spec, DocumentTrainingRequestSpec):
            raise ValueError("Evaluate requires a document Train request for the selected terminal model.")
        return TerminalTrainSource(decoded.id, terminal, decoded.spec)

    return read


def imported_model_reader(store_path: Path) -> Callable[[LoadedModelRead], ModelDocument]:
    @cache
    def provider():
        from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

        return ImmutableArtifactBlobProvider(store_path)

    def read(request: LoadedModelRead) -> ModelDocument:
        loaded = LoadedModelDocument.model_validate_json(request.authored)
        return read_imported_model(provider(), loaded.source)[0]

    return read


def read_imported_model(
    store: ImmutableArtifactBlobProvider, source: ImportedModelSource, *, verify_bytes: bool = False,
) -> tuple[ModelDocument, ModelTree]:
    model = decode_imported_model_document(store.get_object(source.model))
    tree = read_model_tree(store, source.weights, source.encoding)
    for member in tree.members.values():
        if verify_bytes:
            store.verify_object(member)
        else:
            with store.open_owned_object(member):
                pass  # Observe custody and size without reading tensor data during Plan.
    return model, tree


def imported_model_provenance(source: ImportedModelSource, tree: ModelTree) -> ImportedModelProvenance:
    return ImportedModelProvenance(
        kind="imported", model_revision=source.model.sha256, weights_sha256=source.weights.descriptor.sha256,
        replicas=tuple(member.descriptor.id for member in tree.replicas),
    )

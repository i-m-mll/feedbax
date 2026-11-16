"""Read pinned Evaluate subjects from the project's declared Store."""

from collections.abc import Callable
from functools import cache
from pathlib import Path

from feedbax.contracts.project_resolution import TerminalTrainSource


def terminal_train_reader(store_path: Path) -> Callable[[str], TerminalTrainSource | None]:
    @cache
    def provider():
        from feedbax.persistence.artifact_custody import ImmutableArtifactBlobProvider

        return ImmutableArtifactBlobProvider(store_path)

    def read(run: str) -> TerminalTrainSource | None:
        from feedbax.contracts.artifact_custody import ContentTreeRef
        from feedbax.contracts.effective_success import read_success_object, read_success_request, success_content_ref
        from feedbax.contracts.graph import GraphSpec
        from feedbax.contracts.model_checkpoint import MODEL_TREE_GRAPH_PATH
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
        tree = success_content_ref(terminal.model.content)
        if not isinstance(tree, ContentTreeRef):
            raise ValueError("The selected Train model is not a semantic model tree.")
        graph = next(
            (file.object for file in store.get_tree(tree).files if file.relative_path == MODEL_TREE_GRAPH_PATH),
            None,
        )
        if graph is None:
            raise ValueError("The selected Train model tree is missing its graph.")
        return TerminalTrainSource(decoded.id, terminal, decoded.spec, GraphSpec.model_validate_json(store.get_object(graph)))

    return read

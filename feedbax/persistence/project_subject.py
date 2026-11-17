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

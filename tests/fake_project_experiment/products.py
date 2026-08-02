"""The fake project's compiled outputs: real locks over real Feedbax specs.

:mod:`tests.fake_project_experiment` proves ``quillon`` can drive the generic
compile engine. This module proves the *other* end of the same pipe: what a
compile leaves on disk, and what fulfillment reads back from it.

Every output here is emitted through the production emitter —
:func:`~feedbax.contracts.experiment_compile_lock.build_compile_lock` plus
:func:`~feedbax.contracts.authored_canonical.emit_text` — and every compiled
document is a genuine Feedbax spec, because that is what a compiled experiment
layer product is. The project vocabulary stays invented: ``quillon`` names its
recipes, its envelopes, and its roles, and Feedbax supplies the schemas.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.contracts.authored_canonical import canonical_sha256, emit_text
from feedbax.contracts.experiment_compile_lock import (
    CompileLockInputs,
    CompilerContract,
    CompilerImplementation,
    PlannedProductReference,
    build_compile_lock,
)
from feedbax.contracts.figures import FIGURE_SPEC_SCHEMA_ID, FIGURE_SPEC_SCHEMA_VERSION
from feedbax.contracts.manifest import (
    ANALYSIS_RUN_SPEC_SCHEMA_ID,
    ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
    EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
    EVALUATION_RUN_SPEC_SCHEMA_ID,
    EVALUATION_RUN_SPEC_SCHEMA_VERSION,
    REPORT_SPEC_SCHEMA_ID,
    REPORT_SPEC_SCHEMA_VERSION,
)
from feedbax.contracts.experiment_envelope_dialect import (
    EXPERIMENT_ENVELOPE_SCHEMA_VERSION,
    EXPERIMENT_ENVELOPE_SUFFIX,
)
from feedbax.contracts.run_matrix import TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID

from tests.fake_project_experiment import ENVELOPE_DIRECTORY, OUTPUT_DIRECTORY

#: The one dialect every authored envelope declares, engine-owned since the
#: project compiler seam was closed.
ENVELOPE_SCHEMA = EXPERIMENT_ENVELOPE_SCHEMA_VERSION
ENVELOPE_SUFFIX = EXPERIMENT_ENVELOPE_SUFFIX

#: The recipe types quillon registers. Invented vocabulary, as everything the
#: project names is.
PROBE_TYPE = "quillon.probe"
CONDENSE_TYPE = "quillon.condense"
BULLETIN_TYPE = "quillon.bulletin"

QUILLON_CONTRACT = CompilerContract(
    contract_id="quillon.compiler_contract",
    contract_version="quillon.compiler_contract.v1",
)
QUILLON_IMPLEMENTATION = CompilerImplementation(
    code_unit="tests.fake_project_experiment.products", packages=("feedbax",)
)


@dataclass(frozen=True)
class EmittedProduct:
    """One emitted lock/document pair, as a later reference has to address it."""

    name: str
    envelope_ref: str
    envelope_hash: str
    schema_id: str
    content_hash: str
    lock_path: Path
    document_path: Path


class QuillonOutputs:
    """A quillon repository's compiled output directory, written as a compile does."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.output_directory = root / OUTPUT_DIRECTORY
        self.envelope_directory = root / ENVELOPE_DIRECTORY

    def emit(
        self,
        name: str,
        document: Mapping[str, Any],
        *,
        references: Sequence[Any] = (),
        body: Mapping[str, Any] | None = None,
        issue: str | None = None,
    ) -> EmittedProduct:
        """Write one authored envelope and the two files its compile emits."""
        envelope_ref = f"{ENVELOPE_DIRECTORY}/{name}{ENVELOPE_SUFFIX}"
        envelope = {
            "schema": ENVELOPE_SCHEMA,
            "name": name,
            "layer": str(document["schema_id"]),
            "body": dict(body or {}),
        }
        self.envelope_directory.mkdir(parents=True, exist_ok=True)
        (self.root / envelope_ref).write_text(
            json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        family = str(document["schema_id"])
        lock = build_compile_lock(
            CompileLockInputs(
                envelope_ref=envelope_ref,
                envelope_document=envelope,
                envelope_schema=ENVELOPE_SCHEMA,
                name=name,
                family=family,
                compiled_document=document,
                contract=QUILLON_CONTRACT,
                implementation=QUILLON_IMPLEMENTATION,
                references=references,
                issue=issue,
            )
        )
        self.output_directory.mkdir(parents=True, exist_ok=True)
        lock_path = self.output_directory / f"{name}.compile-lock.json"
        document_path = self.output_directory / f"{name}.{family}.json"
        lock_path.write_text(emit_text(lock), encoding="utf-8")
        document_path.write_text(emit_text(document), encoding="utf-8")
        return EmittedProduct(
            name=name,
            envelope_ref=envelope_ref,
            envelope_hash=str(lock["envelope"]["envelope_hash"]),
            schema_id=family,
            content_hash=canonical_sha256(document),
            lock_path=lock_path,
            document_path=document_path,
        )

    # -- the documents each quillon layer compiles to ---------------------

    def probe(self, name: str, *, references: Sequence[Any] = (), **params: Any) -> EmittedProduct:
        """Emit one single evaluation run."""
        return self.emit(
            name,
            {
                "schema_id": EVALUATION_RUN_SPEC_SCHEMA_ID,
                "schema_version": EVALUATION_RUN_SPEC_SCHEMA_VERSION,
                "evaluation_type": PROBE_TYPE,
                "params": {"stage": name, **params},
            },
            references=references,
        )

    def probe_matrix(
        self, name: str, *, references: Sequence[Any] = (), rows: int = 2
    ) -> EmittedProduct:
        """Emit one evaluation run matrix."""
        return self.emit(
            name,
            {
                "schema_id": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_ID,
                "schema_version": EVALUATION_RUN_MATRIX_SPEC_SCHEMA_VERSION,
                "base": {
                    "schema_id": EVALUATION_RUN_SPEC_SCHEMA_ID,
                    "schema_version": EVALUATION_RUN_SPEC_SCHEMA_VERSION,
                    "evaluation_type": PROBE_TYPE,
                    "params": {"stage": name},
                },
                "rows": [
                    {
                        "row_id": f"{name}-{index}",
                        "deltas": [
                            {"path": "params.stage", "value": f"{name}-{index}"}
                        ],
                    }
                    for index in range(rows)
                ],
            },
            references=references,
        )

    def condensate(
        self, name: str, *, references: Sequence[Any] = (), **params: Any
    ) -> EmittedProduct:
        """Emit one analysis run."""
        return self.emit(
            name,
            {
                "schema_id": ANALYSIS_RUN_SPEC_SCHEMA_ID,
                "schema_version": ANALYSIS_RUN_SPEC_SCHEMA_VERSION,
                "analysis_type": CONDENSE_TYPE,
                "params": {"stage": name, **params},
            },
            references=references,
        )

    def bulletin(
        self, name: str, *, references: Sequence[Any] = (), **params: Any
    ) -> EmittedProduct:
        """Emit one report."""
        return self.emit(
            name,
            {
                "schema_id": REPORT_SPEC_SCHEMA_ID,
                "schema_version": REPORT_SPEC_SCHEMA_VERSION,
                "report_type": BULLETIN_TYPE,
                "params": {"stage": name, **params},
            },
            references=references,
        )

    def plate(self, name: str, *, references: Sequence[Any] = ()) -> EmittedProduct:
        """Emit one figure."""
        return self.emit(
            name,
            {
                "schema_id": FIGURE_SPEC_SCHEMA_ID,
                "schema_version": FIGURE_SPEC_SCHEMA_VERSION,
                "name": name,
                "assembler": "feedbax.grid_figure",
            },
            references=references,
        )

    def cohort(self, name: str, *, references: Sequence[Any] = ()) -> EmittedProduct:
        """Emit one training run matrix, which fulfillment never executes."""
        return self.emit(
            name,
            {
                "schema_id": TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID,
                "schema_version": f"{TRAINING_RUN_MATRIX_SPEC_SCHEMA_ID}.v1",
                "name": name,
            },
            references=references,
        )


def planned(
    product: EmittedProduct, *, role_path: str, consumer: Any
) -> PlannedProductReference:
    """Return the planned-product reference one emitted output is addressed by."""
    return PlannedProductReference(
        envelope_ref=product.envelope_ref,
        envelope_hash=product.envelope_hash,
        product_name=product.name,
        product_schema_id=product.schema_id,
        product_schema_version=f"{product.schema_id}.v1",
        compiled_content_hash=product.content_hash,
        role_path=role_path,
        consumer=consumer,
    )


__all__ = [
    "BULLETIN_TYPE",
    "CONDENSE_TYPE",
    "ENVELOPE_DIRECTORY",
    "ENVELOPE_SUFFIX",
    "OUTPUT_DIRECTORY",
    "PROBE_TYPE",
    "QUILLON_CONTRACT",
    "QUILLON_IMPLEMENTATION",
    "EmittedProduct",
    "QuillonOutputs",
    "planned",
]

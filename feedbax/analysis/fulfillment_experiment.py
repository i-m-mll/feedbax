"""Fulfilling one compiled experiment envelope, end to end.

This is the public entry point of the fulfillment lane: given a directory of
compiled experiment outputs and the name of one artifact in it, read the outputs,
derive the plan the target's compile lock implies, prove the closure's boundary,
and walk it — reusing every node whose receipt admits and executing only the ones
that are missing.

There is one argument for what to fulfill and one for where to fulfill it. No
callable is passed in, because every decision the walk makes is Feedbax's:
:mod:`feedbax.analysis.fulfillment_derivation` decides the plan from typed lock
references, and :mod:`feedbax.analysis.fulfillment_lowering` decides the node
request from the compiled document's own schema identity.

``python -m feedbax fulfill-experiment-envelope`` is the same operation from a
shell, with the documented exit codes: ``0`` fulfilled, ``2`` a stable typed
rejection with an actionable diagnostic, ``1`` an infrastructure failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment, FulfillmentRun
from feedbax.analysis.fulfillment_derivation import (
    CompiledOutputIndex,
    derive_fulfillment_plan,
    read_compiled_outputs,
)
from feedbax.analysis.fulfillment_driver import FulfillmentClosure, fulfill_closure, preflight
from feedbax.analysis.fulfillment_plan import FulfillmentPlan


@dataclass(frozen=True)
class ExperimentFulfillment:
    """One fulfilled target: the plan it ran, and what the walk did.

    Attributes:
        target: The target as it was named, before resolution.
        plan: The derived plan, whose ``document()`` is the emittable record of
            what the closure was.
        closure: The preflighted closure, in its one deterministic order.
        run: The walk's per-node results, in execution order.
    """

    target: str
    plan: FulfillmentPlan
    closure: FulfillmentClosure
    run: FulfillmentRun

    def summary(self) -> dict[str, Any]:
        """Return the deterministic record of one fulfillment, for a caller or a shell."""
        return {
            "target": self.plan.target.text,
            "order": list(self.closure.order),
            "executed": list(self.run.executed),
            "reused": list(self.run.reused),
            "receipts": [
                {
                    "node_key": result.node_key,
                    "node_kind": result.node_kind,
                    "disposition": result.disposition,
                    "manifest_ids": [receipt.manifest_id for receipt in result.receipts],
                }
                for result in self.run.results
            ],
        }


def plan_experiment_envelope(
    target: str, *, output_directory: Path | str
) -> tuple[FulfillmentPlan, CompiledOutputIndex]:
    """Derive one target's plan from a directory of compiled experiment outputs.

    Nothing executes and nothing is written. This is the read-only half of
    :func:`fulfill_experiment_envelope`, so a caller can inspect or emit the plan
    without a receipt root.
    """
    index = read_compiled_outputs(output_directory)
    return derive_fulfillment_plan(index, target=target), index


def fulfill_experiment_envelope(
    target: str,
    *,
    output_directory: Path | str,
    environment: FulfillmentEnvironment,
) -> ExperimentFulfillment:
    """Fulfill one compiled experiment target's whole dependency closure.

    Args:
        target: The envelope path or compiled name of the artifact to fulfill.
        output_directory: Where the compile emitted its locks and documents.
        environment: The receipt root, sealed registries, and staged context the
            walk executes against.

    Raises:
        FulfillmentDerivationError: The outputs cannot be read, or the plan
            cannot be derived from what they state.
        ExternalBoundaryError: The closure needs a receipt another entrypoint
            must produce, such as a compiled training run matrix.
        FulfillmentAdmissionError: A stored receipt exists but fails admission.
    """
    plan, index = plan_experiment_envelope(target, output_directory=output_directory)
    closure = preflight(plan, index)
    run = fulfill_closure(closure, environment=environment)
    return ExperimentFulfillment(target=target, plan=plan, closure=closure, run=run)


__all__ = [
    "ExperimentFulfillment",
    "fulfill_experiment_envelope",
    "plan_experiment_envelope",
]

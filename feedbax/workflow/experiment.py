"""Executing one compiled experiment workflow, end to end.

This is the public entry point of the finite workflow lane: given a directory of
compiled experiment outputs and the name of one artifact in it, read the outputs,
derive the plan the target's compile lock implies, prove local realizability,
and walk it — reusing every node whose receipt admits and executing only the ones
that are missing.

There is one argument for what to execute and one for where to execute it. No
callable is passed in, because every decision the walk makes is Feedbax's:
:mod:`feedbax.workflow.derivation` decides the plan from typed lock
references, and :mod:`feedbax.workflow.operation_execution` decides the node
request from the compiled document's own schema identity.

``python -m feedbax execute-experiment-workflow`` is the same operation from a
shell, with the documented exit codes: ``0`` executed, ``2`` a stable typed
rejection with an actionable diagnostic, ``1`` an infrastructure failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from feedbax.analysis.fulfillment_adapters import FulfillmentEnvironment, FulfillmentRun
from feedbax.workflow.derivation import (
    CompiledOutputIndex,
    derive_workflow_plan,
    read_compiled_outputs,
)
from feedbax.workflow.execution import PreparedWorkflow, execute_workflow, prepare_workflow
from feedbax.workflow.plan import WorkflowPlan


@dataclass(frozen=True)
class ExecutedExperimentWorkflow:
    """One executed experiment workflow and its materialization results.

    Attributes:
        target: The target as it was named, before resolution.
        plan: The derived plan, whose ``document()`` is the emittable record of
            what the closure was.
        closure: The prepared workflow, in its one deterministic order.
        run: The walk's per-node results, in execution order.
    """

    target: str
    plan: WorkflowPlan
    closure: PreparedWorkflow
    run: FulfillmentRun

    def summary(self) -> dict[str, Any]:
        """Return the deterministic record of one execution."""
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


def plan_experiment_workflow(
    target: str, *, output_directory: Path | str
) -> tuple[WorkflowPlan, CompiledOutputIndex]:
    """Derive one target's plan from a directory of compiled experiment outputs.

    Nothing executes and nothing is written. This is the read-only half of
    :func:`execute_experiment_workflow`, so a caller can inspect or emit the plan
    without a receipt root.
    """
    index = read_compiled_outputs(output_directory)
    return derive_workflow_plan(index, target=target), index


def execute_experiment_workflow(
    target: str,
    *,
    output_directory: Path | str,
    environment: FulfillmentEnvironment,
) -> ExecutedExperimentWorkflow:
    """Execute one compiled experiment target's finite workflow.

    Args:
        target: The envelope path or compiled name of the workflow target.
        output_directory: Where the compile emitted its locks and documents.
        environment: The receipt root, sealed registries, and staged context the
            walk executes against.

    Raises:
        WorkflowDerivationError: The outputs cannot be read, or the plan
            cannot be derived from what they state.
        ExternalOperationError: The plan contains an operation that must cross
            the later invocation boundary.
        FulfillmentAdmissionError: A stored receipt exists but fails admission.
    """
    plan, index = plan_experiment_workflow(target, output_directory=output_directory)
    closure = prepare_workflow(plan, index)
    run = execute_workflow(closure, environment=environment)
    return ExecutedExperimentWorkflow(target=target, plan=plan, closure=closure, run=run)


__all__ = [
    "ExecutedExperimentWorkflow",
    "execute_experiment_workflow",
    "plan_experiment_workflow",
]

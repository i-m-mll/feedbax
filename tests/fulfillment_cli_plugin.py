"""Quillon's recipes, registered through the ordinary plugin bootstrap.

``python -m feedbax fulfill-experiment-envelope`` composes the application before
it walks anything, so the recipes a compiled document names have to arrive the
same way a real project's do: as a plugin the caller imports with ``--plugin``.
"""

from __future__ import annotations

from feedbax.analysis.evaluation import EvaluationRecipeResult
from feedbax.analysis.reports import REPORT_RENDER_ROLE, ReportRecipeResult
from feedbax.contracts.manifest import store_bytes_artifact
from feedbax.plugins import (
    EVALUATION_RECIPES,
    REPORT_RECIPES,
    FamilyRequirement,
    PluginDeclaration,
    PluginRegistration,
)

from tests.fake_project_extension.products import BULLETIN_TYPE, CONDENSE_TYPE, PROBE_TYPE


def probe_recipe(run_spec, root, states_path, execution_context):
    """Write one small evaluation receipt with one stored artifact."""
    del states_path, execution_context
    artifact = store_bytes_artifact(
        f"{run_spec.params.get('stage', '')}\n".encode(),
        root=root,
        role="evaluation_states",
        logical_name="states.bin",
    )
    return EvaluationRecipeResult(
        states=None,
        summary_metrics={"stage": run_spec.params.get("stage", "")},
        artifacts=[artifact],
        metadata={"states_schema": "quillon.states.v1"},
    )


def bulletin_recipe(report_spec, root, inputs):
    """Render one small report over whatever inputs were bound."""
    artifact = store_bytes_artifact(
        f"# {report_spec.report_type}\n".encode(),
        root=root,
        role=REPORT_RENDER_ROLE,
        logical_name="bulletin.md",
        media_type="text/markdown",
        suffix=".md",
    )
    return ReportRecipeResult(artifacts=[artifact], summary={"inputs": len(inputs)})


def _register(context) -> None:
    evaluations = context.registry(EVALUATION_RECIPES)
    evaluations.register(PROBE_TYPE, probe_recipe)
    evaluations.register(CONDENSE_TYPE, probe_recipe)
    context.registry(REPORT_RECIPES).register(BULLETIN_TYPE, bulletin_recipe)


PLUGIN_REGISTRATION = PluginRegistration(
    PluginDeclaration(
        "tests.fulfillment_cli",
        "1.0",
        1,
        families=(
            FamilyRequirement("evaluation_recipes"),
            FamilyRequirement("report_recipes"),
        ),
    ),
    _register,
)

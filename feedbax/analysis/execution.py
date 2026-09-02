"""Manifest-canonical analysis execution."""

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jax_cookbook.tree as jtree
import plotly.graph_objects as go
from jax_cookbook.progress import piter
from jaxtyping import PyTree

from feedbax.analysis._dependencies import compute_dependency_results
from feedbax.analysis.analysis import AbstractAnalysis, logger
from feedbax.analysis.context import AnalysisRunContext
from feedbax.analysis.types import AnalysisInputData
from feedbax.config import PATHS
from feedbax.config.yaml import get_yaml_loader


@dataclass
class FigDumpManager:
    """Helper for batch-aware figure organization."""

    root: Path = PATHS.figures_dump
    clear_existing: Literal["none", "module"] = "module"
    _counters: dict[str, int] = field(default_factory=dict)

    def module_dir(self, module_key: str) -> Path:
        return self.root / Path(module_key.replace(".", "/"))

    def prepare_module_dir(self, module_key: str, module_config: dict) -> Path:
        directory = self.module_dir(module_key)
        if self.clear_existing in ("module", "all") and directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
        yaml = get_yaml_loader(typ="safe")
        with open(directory / "module.yml", "w") as file:
            yaml.dump(module_config, file)
        return directory

    def prepare_run_dir(self, module_key: str, run_params: dict) -> Path:
        ordinal = self._counters.get(module_key, 0) + 1
        self._counters[module_key] = ordinal
        module_dir = self.module_dir(module_key)
        ordinal_text = f"{ordinal:03d}"
        dump_dir = module_dir / ordinal_text
        dump_dir.mkdir(parents=True, exist_ok=True)
        yaml = get_yaml_loader(typ="safe")
        with open(module_dir / f"{ordinal_text}.yml", "w") as file:
            yaml.dump(run_params, file)
        return dump_dir

    def clear_all_figures(self) -> None:
        """Clear all figures in the root dump directory."""
        if self.root.exists():
            shutil.rmtree(self.root)
            self.root.mkdir(parents=True, exist_ok=True)
            logger.info("Deleted all existing dump figures in %s", self.root)


def perform_all_analyses(
    analyses: dict[str, AbstractAnalysis],
    data: AnalysisInputData,
    *,
    analysis_context: AnalysisRunContext,
    fig_dump_path: Path | None = None,
    fig_dump_formats: list[str] = ["html"],
    custom_dependencies: dict[str, AbstractAnalysis] | None = None,
    requested_outputs: set[str] | None = None,
    **kwargs: Any,
) -> tuple[PyTree[AbstractAnalysis], PyTree[Any], PyTree[go.Figure]]:
    """Run analyses and save their outputs through the canonical manifest context."""
    if not analyses:
        logger.warning("No analyses given to perform; nothing returned")
        return None, None, None

    if not all(isinstance(value, AbstractAnalysis) for value in analyses.values()):
        raise ValueError(
            "All analyses defined in given analysis module must be instances of `AbstractAnalysis`"
        )

    logger.info("Computing results for analyses and their dependencies")
    all_dependency_results = compute_dependency_results(
        analyses,
        data,
        custom_dependencies,
        requested_outputs=requested_outputs,
        analysis_context=analysis_context,
        **kwargs,
    )

    if requested_outputs is not None:
        analyses = {key: value for key, value in analyses.items() if key in requested_outputs}

    def finish_analysis(
        analysis_key: str,
        analysis: AbstractAnalysis,
        inputs: dict[str, Any],
    ) -> tuple[AbstractAnalysis, Any, PyTree[go.Figure] | None]:
        result = inputs.pop("result")
        if "make_figs" not in analysis.__class__.__dict__:
            logger.debug(
                "Skipping figure generation for %s (no make_figs implementation)",
                analysis_key,
            )
            figures = None
        else:
            logger.debug("Making figures: %s", analysis_key)
            figures = analysis._make_figs_with_ops(data, result, **inputs)
            analysis.save_outputs(
                analysis_context,
                result,
                figures,
                data.hps,
                dump_path=fig_dump_path,
                dump_formats=fig_dump_formats,
                label=analysis_key,
                **inputs,
            )
        return analysis, result, figures

    logger.info("Making and saving figures for analyses")
    all_analyses, all_results, all_figures = jtree.unzip(
        {
            analysis_key: finish_analysis(analysis_key, analysis, dependencies)
            for (analysis_key, analysis), dependencies in piter(
                zip(analyses.items(), all_dependency_results),
                total=len(analyses),
                description="Making and saving figures",
                right=lambda progress, _index: progress[0][0],
                eta_halflife=10.0,
            )
        }
    )

    analysis_context.finalize(summary_metrics={"analysis_count": len(analyses)})
    return all_analyses, all_results, all_figures


def run_analyses_with_context(
    analyses: dict[str, AbstractAnalysis],
    data: AnalysisInputData,
    context: AnalysisRunContext,
    *,
    fig_dump_path: Path | None = None,
    fig_dump_formats: list[str] = ["html"],
    custom_dependencies: dict[str, AbstractAnalysis] | None = None,
    requested_outputs: set[str] | None = None,
    **common_inputs: Any,
) -> tuple[PyTree[AbstractAnalysis], PyTree[Any], PyTree[go.Figure]]:
    """Run analyses and write outputs through ``AnalysisRunContext``."""
    return perform_all_analyses(
        analyses,
        data,
        analysis_context=context,
        fig_dump_path=fig_dump_path,
        fig_dump_formats=fig_dump_formats,
        custom_dependencies=custom_dependencies,
        requested_outputs=requested_outputs,
        **common_inputs,
    )

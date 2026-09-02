"""Import-boundary checks for the decomposed analysis module."""

from feedbax.analysis import fig_ops, inputs, result_cache
from feedbax.analysis import analysis as legacy_analysis


def test_analysis_module_reexports_input_dsl_types() -> None:
    assert legacy_analysis.AnalysisRef is inputs.AnalysisRef
    assert legacy_analysis.AbstractAnalysisPorts is inputs.AbstractAnalysisPorts
    assert legacy_analysis.NoPorts is inputs.NoPorts
    assert legacy_analysis.SinglePort is inputs.SinglePort
    assert legacy_analysis.Data is inputs.Data
    assert legacy_analysis.ExpandTo is inputs.ExpandTo
    assert legacy_analysis.Transformed is inputs.Transformed
    assert legacy_analysis.LiteralInput is inputs.LiteralInput
    assert legacy_analysis.CallWithDeps is inputs.CallWithDeps


def test_analysis_module_reexports_operation_helpers() -> None:
    assert legacy_analysis.FigIterCtx is fig_ops.FigIterCtx
    assert legacy_analysis.FigureSaveTask is fig_ops.FigureSaveTask
    assert legacy_analysis._PrepOp is fig_ops._PrepOp
    assert legacy_analysis._FigOp is fig_ops._FigOp
    assert legacy_analysis._FinalOp is fig_ops._FinalOp
    assert legacy_analysis._AnalysisVmapSpec is fig_ops._AnalysisVmapSpec
    assert legacy_analysis._apply_fig_ops is fig_ops._apply_fig_ops
    assert legacy_analysis._apply_final_ops is fig_ops._apply_final_ops


def test_analysis_module_reexports_result_cache_helpers() -> None:
    assert legacy_analysis.RESULTS_CACHE_SUBDIR is result_cache.RESULTS_CACHE_SUBDIR
    assert (
        legacy_analysis.ANALYSIS_RESULT_CACHE_SCHEMA_VERSION
        is result_cache.ANALYSIS_RESULT_CACHE_SCHEMA_VERSION
    )
    assert legacy_analysis.AnalysisResultCacheCorruption is result_cache.AnalysisResultCacheCorruption
    assert legacy_analysis._load_analysis_result_cache is result_cache._load_analysis_result_cache
    assert legacy_analysis._write_analysis_result_cache is result_cache._write_analysis_result_cache

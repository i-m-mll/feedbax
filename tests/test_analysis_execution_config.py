from __future__ import annotations

from types import SimpleNamespace

import pytest

import feedbax.analysis.execution as execution
from feedbax.analysis.execution import AnalysisModelLoadConfig
from feedbax.config.namespace import TreeNamespace


def test_analysis_module_must_supply_model_load_config() -> None:
    with pytest.raises(ValueError, match="MODEL_LOAD_CONFIG"):
        execution._required_model_load_config("demo.analysis", SimpleNamespace())


def test_model_load_config_supplies_sweep_label_and_query(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_query_and_load_model(*args, params_query, **kwargs):
        calls.append(params_query)
        value = params_query["condition"]
        return (
            (f"task-{value}", f"model-{value}"),
            f"record-{value}",
            {"included": value},
            1,
            TreeNamespace(source=value),
        )

    monkeypatch.setattr(execution, "query_and_load_model", fake_query_and_load_model)
    registry = SimpleNamespace(
        get_training_module=lambda name: SimpleNamespace(setup_task_model_pair=object())
    )
    hps = TreeNamespace(conditions=("small", "large"))
    config = AnalysisModelLoadConfig(
        training_module_name="project.training",
        sweep_label="condition",
        sweep_values=lambda hps: hps.conditions,
        params_query=lambda hps, value, module: {
            "expt_name": module,
            "condition": value,
        },
    )

    execution.load_trained_models_and_aux_objects(
        config,
        "project.training",
        hps,
        SimpleNamespace(),
        registry,
    )

    assert calls == [
        {"expt_name": "project.training", "condition": "small"},
        {"expt_name": "project.training", "condition": "large"},
    ]

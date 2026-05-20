"""Tests for loss-history plotting helpers."""

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from feedbax.loss import TermTree
from feedbax.plot import loss_history, loss_history_compare


def _loss_tree(scale: float = 1.0, *, include_effort: bool = True) -> TermTree:
    terms = {
        "reach": TermTree.leaf(
            "reach",
            scale
            * jnp.asarray(
                [
                    [4.0, 6.0],
                    [2.0, 3.0],
                    [1.0, 1.5],
                ]
            ),
        ),
    }
    if include_effort:
        terms["effort"] = TermTree.leaf(
            "effort",
            scale
            * jnp.asarray(
                [
                    [2.0, 2.5],
                    [1.0, 1.25],
                    [0.5, 0.75],
                ]
            ),
        )

    return TermTree.branch("loss", terms)


def _history(scale: float = 1.0, *, include_effort: bool = True) -> SimpleNamespace:
    losses = _loss_tree(scale, include_effort=include_effort)
    return SimpleNamespace(loss=losses, loss_validation=losses)


def test_loss_history_still_plots_single_run_terms() -> None:
    fig = loss_history(_loss_tree())

    assert [trace.name for trace in fig.data if trace.showlegend is not False] == [
        "effort",
        "reach",
        "Total",
    ]
    assert len(fig.data) == 9
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"


def test_loss_history_accepts_raw_loss_array() -> None:
    fig = loss_history(jnp.asarray([[3.0, 4.0], [2.0, 3.0], [1.0, 2.0]]))

    assert [trace.name for trace in fig.data if trace.showlegend is not False] == ["Total"]
    assert len(fig.data) == 3


def test_loss_history_compare_plots_runs_on_shared_term_axes() -> None:
    fig = loss_history_compare(
        {
            "baseline": _history(1.0),
            "variant": _history(0.5),
        },
        n_cols=2,
    )

    mean_traces = [trace for trace in fig.data if trace.showlegend is not False]
    assert [trace.name for trace in mean_traces] == ["baseline", "variant"]
    assert len(fig.data) == 18
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"
    assert fig.layout.xaxis2.type == "log"
    assert fig.layout.yaxis2.type == "log"


def test_loss_history_compare_uses_shared_terms_by_default() -> None:
    fig = loss_history_compare(
        {
            "baseline": _history(1.0),
            "variant": _history(0.5, include_effort=False),
        },
    )

    assert len(fig.layout.annotations) == 2
    assert [annotation.text for annotation in fig.layout.annotations] == ["Total", "reach"]
    assert len(fig.data) == 12


def test_loss_history_compare_rejects_explicit_missing_terms() -> None:
    with pytest.raises(ValueError, match="missing from histories"):
        loss_history_compare(
            {
                "baseline": _history(1.0),
                "variant": _history(0.5, include_effort=False),
            },
            terms=["Total", "effort"],
        )

# pyright: reportMissingTypeStubs=false
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from feedbax.misc import batch_reshape, nan_bypass

ArrayFunc = Callable[[Any], Any]


def simple_transform(x):
    return 2 * x + 1


def pca_like_transform(x):
    if bool(np.asarray(np.isnan(x).any())):
        raise ValueError("Input contains NaN values")
    return x[..., :2]


def batch_reshape_with_nan_bypass(func: ArrayFunc) -> ArrayFunc:
    return cast(ArrayFunc, batch_reshape(cast(ArrayFunc, nan_bypass(func))))


def test_batch_reshape_composes_with_nan_bypass():
    data = np.arange(60, dtype=float).reshape(3, 4, 5)
    data[0, 1, :] = np.nan
    data[1, 2, 2:] = np.nan
    data[2, 0, :] = np.nan

    transform = batch_reshape_with_nan_bypass(simple_transform)
    result = np.asarray(transform(data))

    nan_rows = np.isnan(data).reshape(-1, data.shape[-1]).any(axis=1)
    result_flat = result.reshape(-1, result.shape[-1])

    assert result.shape == data.shape
    assert np.isnan(result_flat[nan_rows]).all()
    np.testing.assert_allclose(
        result_flat[~nan_rows],
        2 * data.reshape(-1, 5)[~nan_rows] + 1,
        rtol=1e-6,
    )


def test_nan_bypass_all_nan_input():
    data = np.full((2, 3, 4), np.nan)

    result = np.asarray(batch_reshape_with_nan_bypass(simple_transform)(data))

    assert result.shape == data.shape
    assert np.isnan(result).all()


def test_nan_bypass_no_nan_input():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(2, 3, 4))

    result = np.asarray(batch_reshape_with_nan_bypass(simple_transform)(data))

    assert result.shape == data.shape
    np.testing.assert_allclose(result, 2 * data + 1, rtol=1e-6)


def test_pca_like_transform_rejects_nan_without_bypass():
    data = np.arange(30, dtype=float).reshape(2, 3, 5)
    data[0, 1, :] = np.nan

    transform = batch_reshape(pca_like_transform)

    with np.testing.assert_raises(ValueError):
        transform(data)


def test_pca_like_transform_preserves_nan_rows_with_bypass():
    data = np.arange(30, dtype=float).reshape(2, 3, 5)
    data[0, 1, :] = np.nan
    data[1, 0, 2:] = np.nan

    transform = batch_reshape_with_nan_bypass(pca_like_transform)
    result = np.asarray(transform(data))

    nan_rows = np.isnan(data).reshape(-1, data.shape[-1]).any(axis=1)
    result_flat = result.reshape(-1, result.shape[-1])

    assert result.shape == (2, 3, 2)
    assert np.isnan(result_flat[nan_rows]).all()
    np.testing.assert_allclose(result_flat[~nan_rows], data.reshape(-1, 5)[~nan_rows, :2])

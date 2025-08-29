from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from napari_prism.models.adata_ops.cell_typing._preprocessing import (
    _current_backend,
    fill_na,
    filter_by_obs_count,
    filter_by_obs_quantile,
    filter_by_obs_value,
    filter_by_var_quantile,
    filter_by_var_value,
    set_backend,
    split_by_obs,
    trim_kwargs,
    with_current_backend,
)


def test_set_backend_cpu():
    set_backend("cpu")
    assert _current_backend["module"] == "scanpy"


def test_set_backend_invalid():
    with pytest.raises(
        ValueError, match="Invalid backend. Must be 'cpu' or 'gpu'"
    ):
        set_backend("invalid")


def test_trim_kwargs():
    def dummy_function(a, b, c=None):
        pass

    kwargs = {"a": 1, "b": 2, "c": 3, "d": 4}
    trimmed = trim_kwargs(kwargs, dummy_function)
    expected = {"a": 1, "b": 2, "c": 3}
    assert trimmed == expected


def test_filter_by_obs_count_categorical(adata):
    # Add a categorical column
    adata.obs["test_cat"] = pd.Categorical(
        ["A"] * 300 + ["B"] * 200 + ["C"] * 200
    )

    result = filter_by_obs_count(adata, "test_cat", min_value=250, copy=True)

    # Should only keep category A (300 cells)
    assert result.n_obs == 300
    assert all(result.obs["test_cat"] == "A")


def test_filter_by_obs_count_invalid_column(adata):
    with pytest.raises(ValueError, match="Must be a categorical .obs column"):
        filter_by_obs_count(adata, "n_genes", min_value=10, copy=True)


def test_filter_by_obs_value_numeric(adata):
    result = filter_by_obs_value(
        adata, "n_genes", min_value=100, max_value=2000, copy=True
    )

    assert all(result.obs["n_genes"] > 100)
    assert all(result.obs["n_genes"] < 2000)
    assert result.n_obs <= adata.n_obs


def test_filter_by_obs_value_invalid_column(adata):
    adata.obs["test_cat"] = pd.Categorical(["A"] * adata.n_obs)

    with pytest.raises(ValueError, match="Must be a numerical .obs column"):
        filter_by_obs_value(adata, "test_cat", min_value=10, copy=True)


def test_filter_by_obs_quantile(adata):
    result = filter_by_obs_quantile(
        adata, "n_genes", min_quantile=0.1, max_quantile=0.9, copy=True
    )

    min_val = np.quantile(adata.obs["n_genes"], 0.1)
    max_val = np.quantile(adata.obs["n_genes"], 0.9)

    assert result.obs["n_genes"].min() > min_val
    assert result.obs["n_genes"].max() < max_val


def test_filter_by_var_value(adata):
    """Test filtering by var value."""
    var_name = adata.var_names[0]  # Get first gene
    result = filter_by_var_value(
        adata, var_name, min_value=0.1, max_value=10, copy=True
    )

    assert result.n_obs <= adata.n_obs


def test_filter_by_var_quantile(adata):
    """Test filtering by var quantile."""
    var_name = adata.var_names[0]  # Get first gene
    result = filter_by_var_quantile(
        adata, var_name, min_quantile=0.1, max_quantile=0.9, copy=True
    )

    assert result.n_obs <= adata.n_obs


def test_fill_na(adata):
    """Test filling NaN values."""
    # Add some NaN values
    adata.X[0, 0] = np.nan
    adata.X[1, 1] = np.nan

    result = fill_na(adata, fill_value=0.0, copy=True)

    assert not np.any(np.isnan(result.X))


def test_fill_na_layer(adata):
    """Test filling NaN values in a layer."""
    # Add a layer with NaN values
    adata.layers["test"] = adata.X.copy()
    adata.layers["test"][0, 0] = np.nan

    result = fill_na(adata, fill_value=0.0, layer="test", copy=True)

    assert not np.any(np.isnan(result.layers["test"]))


def test_split_by_obs(adata):
    # categorical column for splitting
    adata.obs["split_col"] = pd.Categorical(["A"] * 300 + ["B"] * 400)

    adata_list, obs_var = split_by_obs(adata, "split_col")

    assert len(adata_list) == 2
    assert obs_var == "split_col"
    assert adata_list[0].n_obs == 300
    assert adata_list[1].n_obs == 400


def test_split_by_obs_with_selections(adata):
    adata.obs["split_col"] = pd.Categorical(
        ["A"] * 200 + ["B"] * 250 + ["C"] * 250
    )

    adata_list, obs_var = split_by_obs(
        adata, "split_col", selections=["A", "C"]
    )

    assert len(adata_list) == 2
    assert obs_var == "split_col"
    assert adata_list[0].n_obs == 200  # A
    assert adata_list[1].n_obs == 250  # C


def test_with_current_backend_decorator():
    @with_current_backend
    def dummy_function(adata, param1=None):
        return adata

    # decorator expected to preserve function behavior
    result = dummy_function(MagicMock(), param1="test")
    assert result is not None

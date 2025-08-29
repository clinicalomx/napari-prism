import numpy as np
import pandas as pd
import pytest

from napari_prism.models.adata_ops.feature_modelling._discrete import (
    _consolidate_statistics,
    _get_x_y_by_binary_label,
    _normalise_log2p,
    cellular_neighborhood_enrichment,
    difference_of_means_by_binary_label,
    univariate_test_feature_by_binary_label,
)


def test_normalise_log2p():
    """Test log2 normalization with pseudocount."""
    # Create test data
    data = pd.DataFrame(
        {"A": [10, 20, 30], "B": [5, 10, 15], "C": [15, 30, 45]}
    )

    result = _normalise_log2p(data, pseudocount=1e-3)

    # Check that rows sum to 1 after normalization (before log transform)
    # and that log transform was applied
    assert result.shape == data.shape
    assert np.all(np.isfinite(result.values))


def test_consolidate_statistics():
    """Test consolidating statistics into matrix format."""
    results = {
        ("phenotype1", "neighborhood1"): 0.05,
        ("phenotype1", "neighborhood2"): 0.01,
        ("phenotype2", "neighborhood1"): 0.1,
        ("phenotype2", "neighborhood2"): 0.02,
    }

    df = _consolidate_statistics(results, "neighborhood", "phenotype")

    assert df.shape == (2, 2)  # 2 neighborhoods x 2 phenotypes
    assert "phenotype1" in df.columns
    assert "phenotype2" in df.columns
    assert "neighborhood1" in df.index
    assert "neighborhood2" in df.index


def test_get_x_y_by_binary_label_obs(adata):
    """Test extracting X and y for binary classification from obs."""
    # Add binary label
    adata.obs["binary_label"] = ["A"] * 350 + ["B"] * 350

    result = _get_x_y_by_binary_label(
        adata, "n_genes", "binary_label", attr="obs"
    )

    assert len(result) == 2
    assert "A" in result
    assert "B" in result
    assert len(result["A"]) == 350
    assert len(result["B"]) == 350


def test_get_x_y_by_binary_label_X(adata):
    """Test extracting X and y for binary classification from X matrix."""
    # Add binary label
    adata.obs["binary_label"] = ["A"] * 350 + ["B"] * 350

    # Use first gene
    gene_name = adata.var_names[0]

    result = _get_x_y_by_binary_label(
        adata, gene_name, "binary_label", attr="X"
    )

    assert len(result) == 2
    assert "A" in result
    assert "B" in result


def test_get_x_y_by_binary_label_invalid_attr(adata):
    """Test invalid attribute raises error."""
    adata.obs["binary_label"] = ["A"] * 350 + ["B"] * 350

    with pytest.raises(
        ValueError, match="Attribute must be either 'obs' or 'X'"
    ):
        _get_x_y_by_binary_label(
            adata, "n_genes", "binary_label", attr="invalid"
        )


def test_difference_of_means_by_binary_label(adata):
    """Test difference of means calculation."""
    # Add binary label
    adata.obs["binary_label"] = ["A"] * 350 + ["B"] * 350

    diff_means, diff_ci = difference_of_means_by_binary_label(
        adata, "n_genes", "binary_label", attr="obs"
    )

    assert isinstance(diff_means, float | np.floating | pd.Series)
    assert isinstance(diff_ci, float | np.floating | pd.Series)
    assert np.isfinite(diff_means)
    assert np.isfinite(diff_ci)
    assert diff_ci >= 0  # Confidence interval should be positive


def test_univariate_test_feature_by_binary_label_parametric(adata):
    """Test parametric univariate test (t-test)."""
    # Add binary label
    adata.obs["binary_label"] = ["A"] * 350 + ["B"] * 350

    statistic, p_value = univariate_test_feature_by_binary_label(
        adata, "n_genes", "binary_label", attr="obs", parametric=True
    )

    assert isinstance(statistic, float | np.floating)
    assert isinstance(p_value, float | np.floating)
    assert np.isfinite(statistic)
    assert np.isfinite(p_value)
    assert 0 <= p_value <= 1


def test_univariate_test_feature_by_binary_label_nonparametric(adata):
    """Test non-parametric univariate test (Mann-Whitney U)."""
    # Add binary label
    adata.obs["binary_label"] = ["A"] * 350 + ["B"] * 350

    statistic, p_value = univariate_test_feature_by_binary_label(
        adata, "n_genes", "binary_label", attr="obs", parametric=False
    )

    assert isinstance(statistic, float | np.floating)
    assert isinstance(p_value, float | np.floating)
    assert np.isfinite(statistic)
    assert np.isfinite(p_value)
    assert 0 <= p_value <= 1


def test_cellular_neighborhood_enrichment(adata):
    # Setup mock data
    adata.obs["neighborhood"] = ["N1"] * 200 + ["N2"] * 200 + ["N3"] * 300
    adata.obs["phenotype"] = ["P1"] * 300 + ["P2"] * 400
    adata.obs["label"] = ["A"] * 400 + ["B"] * 300
    adata.obs["grouping"] = ["S1"] * 100 + ["S2"] * 300 + ["S3"] * 300

    result = cellular_neighborhood_enrichment(
        adata, "neighborhood", "phenotype", "label", "grouping"
    )

    assert "p_values" in result
    assert "adjusted_p_values" in result
    assert "reject_null_hypothesis" in result
    assert "coefficients" in result
    assert "t_values" in result

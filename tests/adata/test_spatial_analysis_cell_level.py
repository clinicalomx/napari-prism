from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from napari_prism.models.adata_ops.spatial_analysis._cell_level import (
    _get_neighborhoods_from_job,
    _sort_neighbors,
    cellular_neighborhoods_sq,
    compute_pair_interactions,
    compute_targeted_degree_ratio,
    proximity_density,
)


def test_compute_targeted_degree_ratio(adata):
    # Add phenotype column
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    # Create a simple adjacency matrix
    n_cells = adata.n_obs
    adjacency_matrix = np.random.randint(0, 2, size=(n_cells, n_cells))
    np.fill_diagonal(adjacency_matrix, 0)  # No self-connections

    result = compute_targeted_degree_ratio(
        adata, adjacency_matrix, "phenotype", "A", "B", directed=True
    )

    assert len(result) == 300  # Should return ratio for each A cell
    assert np.all(result >= 0)  # Ratios should be non-negative
    assert np.all(result <= 1)  # Ratios should be <= 1


def test_compute_targeted_degree_ratio_undirected(adata):
    # Add phenotype column
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    # Create a simple adjacency matrix
    n_cells = adata.n_obs
    adjacency_matrix = np.random.randint(0, 2, size=(n_cells, n_cells))
    np.fill_diagonal(adjacency_matrix, 0)  # No self-connections

    result = compute_targeted_degree_ratio(
        adata, adjacency_matrix, "phenotype", "A", "B", directed=False
    )

    assert len(result) == 300  # Should return ratio for each A cell
    assert np.all(result >= 0)  # Ratios should be non-negative
    assert np.all(result <= 1)  # Ratios should be <= 1


def test_compute_pair_interactions_nodes(adata):
    # Add phenotype column and reset index
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    # Create a simple sparse adjacency matrix
    n_cells = adata.n_obs
    adjacency_matrix = csr_matrix(
        np.random.randint(0, 2, size=(n_cells, n_cells))
    )

    # Add to obsp
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    total_interactions, total_cells, missing = compute_pair_interactions(
        adata, "phenotype", "A", "B", method="nodes"
    )

    assert isinstance(
        total_interactions, int | float | np.integer | np.floating
    )
    assert isinstance(total_cells, int | np.integer)
    assert isinstance(missing, bool)
    assert total_interactions >= 0
    assert total_cells > 0


def test_compute_pair_interactions_edges(adata):
    # Add phenotype column
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    # Create a simple sparse adjacency matrix
    n_cells = adata.n_obs
    adjacency_matrix = csr_matrix(
        np.random.randint(0, 2, size=(n_cells, n_cells))
    )

    # Add to obsp
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    total_interactions, total_cells, missing = compute_pair_interactions(
        adata, "phenotype", "A", "B", method="edges"
    )

    assert isinstance(
        total_interactions, int | float | np.integer | np.floating
    )
    assert isinstance(total_cells, int | np.integer)
    assert isinstance(missing, bool)
    assert total_interactions >= 0
    assert total_cells > 0


def test_compute_pair_interactions_same_phenotype(adata):
    # Add phenotype column
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    # Create a simple sparse adjacency matrix
    n_cells = adata.n_obs
    adjacency_matrix = csr_matrix(
        np.random.randint(0, 2, size=(n_cells, n_cells))
    )

    # Add to obsp
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    total_interactions, total_cells, missing = compute_pair_interactions(
        adata, "phenotype", "A", "A", method="nodes"
    )

    assert isinstance(
        total_interactions, int | float | np.integer | np.floating
    )
    assert isinstance(total_cells, int | float | np.integer | np.floating)
    assert isinstance(missing, bool)


def test_compute_pair_interactions_missing_key(adata):
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    with pytest.raises(ValueError, match="No adjacency matrix provided"):
        compute_pair_interactions(adata, "phenotype", "A", "B", method="nodes")


def test_compute_pair_interactions_invalid_method(adata):
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400
    n_cells = adata.n_obs
    adjacency_matrix = csr_matrix(
        np.random.randint(0, 2, size=(n_cells, n_cells))
    )
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    with pytest.raises(ValueError, match="invalid method"):
        compute_pair_interactions(
            adata, "phenotype", "A", "B", method="invalid"
        )


@patch("napari_prism.models.adata_ops.spatial_analysis._cell_level.Parallel")
def test_proximity_density(mock_parallel, adata):
    # Setup test data
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400
    adata.obs["grouping"] = ["G1"] * 200 + ["G2"] * 250 + ["G3"] * 250

    # Create adjacency matrix
    n_cells = adata.n_obs
    adjacency_matrix = csr_matrix(
        np.random.randint(0, 2, size=(n_cells, n_cells))
    )
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    # Mock parallel processing
    mock_parallel.return_value.return_value = [
        (
            "G1",
            {("A", "A"): 0.1, ("A", "B"): 0.2, ("B", "B"): 0.3},
            {("A", "A"): False, ("A", "B"): False, ("B", "B"): False},
            {("A", "A"): 100, ("A", "B"): 150, ("B", "B"): 120},
        ),
        (
            "G2",
            {("A", "A"): 0.15, ("A", "B"): 0.25, ("B", "B"): 0.35},
            {("A", "A"): False, ("A", "B"): False, ("B", "B"): False},
            {("A", "A"): 110, ("A", "B"): 160, ("B", "B"): 130},
        ),
    ]

    result = proximity_density(adata, "grouping", "phenotype", inplace=False)

    assert len(result) == 3  # Should return 3 dataframes
    grouping_df, mask_df, count_df = result

    assert isinstance(grouping_df, pd.DataFrame)
    assert isinstance(mask_df, pd.DataFrame)
    assert isinstance(count_df, pd.DataFrame)


def test_proximity_density_missing_connectivity(adata):
    """Test error when connectivity key is missing."""
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400
    adata.obs["grouping"] = ["G1"] * 200 + ["G2"] * 250 + ["G3"] * 250

    with pytest.raises(ValueError, match="No adjacency matrix found"):
        proximity_density(adata, "grouping", "phenotype")


def test_proximity_density_missing_grouping(adata):
    """Test error when grouping column is missing."""
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400
    n_cells = adata.n_obs
    adjacency_matrix = csr_matrix(
        np.random.randint(0, 2, size=(n_cells, n_cells))
    )
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    with pytest.raises(ValueError, match="Grouping column not found"):
        proximity_density(adata, "missing_grouping", "phenotype")


def test_sort_neighbors():
    """Test sorting neighbors by distance."""
    # Create mock region with index
    region = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
    region.index = [10, 20, 30]

    # Mock distances and indices from NearestNeighbors
    distances = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
    indices = np.array([[0, 1, 2], [1, 0, 2], [2, 1, 0]])

    result = _sort_neighbors(region, distances, indices)

    assert result.shape == (3, 3)
    assert result.dtype == np.int32 or result.dtype == int


@patch(
    "napari_prism.models.adata_ops.spatial_analysis._cell_level.NearestNeighbors"
)
def test_get_neighborhoods_from_job(mock_nn):
    # Mock data
    region_data = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [1, 2, 3, 4, 5]})
    region_data.index = [10, 20, 30, 40, 50]

    region_groupby = MagicMock()
    region_groupby.get_group.return_value = region_data

    # Mock NearestNeighbors
    mock_nn_instance = MagicMock()
    mock_nn_instance.kneighbors.return_value = (
        np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 2.0], [2.0, 1.0], [1.0, 2.0]]),
        np.array([[0, 1], [1, 0], [2, 1], [3, 2], [4, 3]]),
    )
    mock_nn.return_value.fit.return_value = mock_nn_instance

    job = (None, "region1", [10, 20, 30, 40, 50])

    result = _get_neighborhoods_from_job(
        job, region_groupby, 2, "x", "y", None
    )

    assert result.shape == (5, 2)
    assert result.dtype == np.int32


@patch(
    "napari_prism.models.adata_ops.spatial_analysis._cell_level.MiniBatchKMeans"
)
def test_cellular_neighborhoods_sq(mock_kmeans, adata):
    # Setup test data
    adata.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    # Create adjacency matrix
    n_cells = adata.n_obs
    adjacency_matrix = csr_matrix(
        np.random.randint(0, 2, size=(n_cells, n_cells))
    )
    adata.obsp["spatial_connectivities"] = adjacency_matrix

    # Mock KMeans
    mock_kmeans_instance = MagicMock()
    mock_kmeans_instance.fit_predict.return_value = np.random.randint(
        0, 3, n_cells
    )
    mock_kmeans_instance.cluster_centers_ = np.random.rand(3, 2)
    mock_kmeans_instance.inertia_ = 100.0
    mock_kmeans.return_value = mock_kmeans_instance

    cellular_neighborhoods_sq(
        adata, "phenotype", "spatial_connectivities", k_kmeans=[3, 5]
    )

    # Check that results were stored
    assert "neighbor_counts" in adata.obsm
    assert "cn_enrichment_matrices" in adata.uns
    assert "cn_labels" in adata.obsm
    assert "cn_inertias" in adata.uns

    # Check shapes
    assert adata.obsm["neighbor_counts"].shape[0] == n_cells
    assert adata.obsm["cn_labels"].shape[0] == n_cells
    assert adata.obsm["cn_labels"].shape[1] == 2  # Two k values

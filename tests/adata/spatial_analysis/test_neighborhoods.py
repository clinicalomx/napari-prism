from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from napari_prism.models.adata_ops.spatial_analysis.neighborhoods.nolan import (
    cellular_neighborhoods_sq,
)


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

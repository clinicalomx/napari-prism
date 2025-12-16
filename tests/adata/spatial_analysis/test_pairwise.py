import numpy as np
import pandas as pd

from napari_prism.models.adata_ops.spatial_analysis.pairwise.scimap import (
    compute_pair_interactions,
    compute_targeted_degree_ratio,
    proximity_density,
)


def test_compute_targeted_degree_ratio(adata_spatial_graph):
    # Add phenotype column
    print(adata_spatial_graph)
    adata_spatial_graph.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    adjacency_matrix = adata_spatial_graph.obsp["spatial_connectivities"]
    result = compute_targeted_degree_ratio(
        adata_spatial_graph,
        adjacency_matrix,
        "phenotype",
        "A",
        "B",
        directed=True,
    )

    assert len(result) == 300  # Should return ratio for each A cell
    assert np.all(result >= 0)  # Ratios should be non-negative
    assert np.all(result <= 1)  # Ratios should be <= 1


def test_compute_targeted_degree_ratio_undirected(adata_spatial_graph):
    # Add phenotype column
    adata_spatial_graph.obs["phenotype"] = ["A"] * 300 + ["B"] * 400
    adjacency_matrix = adata_spatial_graph.obsp["spatial_connectivities"]
    result = compute_targeted_degree_ratio(
        adata_spatial_graph,
        adjacency_matrix,
        "phenotype",
        "A",
        "B",
        directed=False,
    )

    assert len(result) == 300  # Should return ratio for each A cell
    assert np.all(result >= 0)  # Ratios should be non-negative
    assert np.all(result <= 1)  # Ratios should be <= 1


def test_compute_pair_interactions_nodes(adata_spatial_graph):
    # Add phenotype column and reset index
    adata_spatial_graph.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    total_interactions, total_cells, missing = compute_pair_interactions(
        adata_spatial_graph, "phenotype", "A", "B", method="nodes"
    )

    assert isinstance(
        total_interactions, int | float | np.integer | np.floating
    )
    assert isinstance(total_cells, int | np.integer)
    assert isinstance(missing, bool)
    assert total_interactions >= 0
    assert total_cells > 0


def test_compute_pair_interactions_edges(adata_spatial_graph):
    # Add phenotype column
    adata_spatial_graph.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    total_interactions, total_cells, missing = compute_pair_interactions(
        adata_spatial_graph, "phenotype", "A", "B", method="edges"
    )

    assert isinstance(
        total_interactions, int | float | np.integer | np.floating
    )
    assert isinstance(total_cells, int | np.integer)
    assert isinstance(missing, bool)
    assert total_interactions >= 0
    assert total_cells > 0


def test_compute_pair_interactions_same_phenotype(adata_spatial_graph):
    # Add phenotype column
    adata_spatial_graph.obs["phenotype"] = ["A"] * 300 + ["B"] * 400

    total_interactions, total_cells, missing = compute_pair_interactions(
        adata_spatial_graph, "phenotype", "A", "A", method="nodes"
    )

    assert isinstance(
        total_interactions, int | float | np.integer | np.floating
    )
    assert isinstance(total_cells, int | float | np.integer | np.floating)
    assert isinstance(missing, bool)


def test_proximity_density(adata_spatial_graph):
    # Setup test data
    adata_spatial_graph.obs["grouping"] = (
        ["G1"] * 200 + ["G2"] * 250 + ["G3"] * 250
    )
    result = proximity_density(
        adata_spatial_graph, "grouping", "bulk_labels", inplace=False
    )

    assert len(result) == 3  # Should return 3 dataframes
    grouping_df, mask_df, count_df = result

    assert isinstance(grouping_df, pd.DataFrame)
    assert isinstance(mask_df, pd.DataFrame)
    assert isinstance(count_df, pd.DataFrame)


# def test_gcross(adata_spatial):
#     results = gcross(
#         adata_spatial, sample_key="library", cell_type_key="bulk_labels"
#     )


# xs, ys = results["s1"][("CD14+ Monocyte", "CD14+ Monocyte")]
# import matplotlib.pyplot as plt
# plt.plot(xs, ys)

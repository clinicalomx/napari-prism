"""Spatial analyses which generate metrics at the cell level."""

from itertools import combinations_with_replacement
from typing import Literal

import numpy as np
import pandas as pd
import scipy
from anndata import AnnData
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix

from napari_prism.models.adata_ops.spatial_analysis.utils import (
    symmetrise_graph,
)


# Pairwise Cell Computations
def compute_targeted_degree_ratio(
    adata,
    adjacency_matrix,
    phenotype_column,
    phenotype_A,  # source phenotype
    phenotype_B,  # target phenotype
    directed=False,
):
    """For each node in the adjacency matrix, compute the ratio of its
    targets that are of phenotype_pair.

    If directed, then this becomes the outdegree ratio. i.e.) If
    KNN, then the score is the ratio of its closest K neighbors being of
    the other specified type.

    If not directed, then this becomes a simple degree ratio, with the graph
    being symmetrised (enforce A->B, then B->A).

    """
    mat = adjacency_matrix if directed else symmetrise_graph(adjacency_matrix)

    a_mask = adata.obs[phenotype_column] == phenotype_A
    b_mask = adata.obs[phenotype_column] == phenotype_B
    a_ix = list(np.where(a_mask)[0])
    b_ix = list(np.where(b_mask)[0])
    a = mat[a_ix]  # A rows -> all cols
    ab = mat[np.ix_(a_ix, b_ix)]  # A rows -> B cols

    a_edge_degrees = a.sum(axis=1)  # Total connections for each A cell
    a_target_degrees = ab.sum(
        axis=1
    )  # Total connections to B cells for each A cell

    a_ab = np.divide(
        a_target_degrees, a_edge_degrees
    )  # For each A cell, ratio of B connections to total connections

    return a_ab


def compute_pair_interactions(
    adata: AnnData,
    phenotype_column: str,
    phenotype_A: str,
    phenotype_B: str,
    method: Literal["nodes", "edges"],
    adjacency_matrix: np.ndarray | scipy.sparse.csr.csr_matrix = None,
    connectivity_key: str = "spatial_connectivities",
) -> tuple[int, int, bool]:
    """
    Uses adjacency_matrix first if supplied, otherwise tries to find
    adjacency matrix in adata.obsp using `connectivity_key`.

    Compute the number of interactions between two phenotypes in a graph.
    Enforced symmetric relations. i.e.) IF A -> B, then B -> A.

    If neighbors graph constructed with radius, then already symmetric.

    Returns:
    total_interactions: Number of interactions between phenotype_pair
    total_cells: Total number of cells in the graph
    missing: True if not enough cells for comparison
    """
    adata = adata.copy()
    adata.obs = adata.obs.reset_index()
    if adjacency_matrix is None:
        if connectivity_key not in adata.obsp:
            raise ValueError(
                "No adjacency matrix provided and no "
                "connectivity key found in adata.obsp"
            )
        else:
            adjacency_matrix = adata.obsp[connectivity_key]

    sym = symmetrise_graph(adjacency_matrix)
    a_ix = list(
        adata.obs[adata.obs[phenotype_column] == phenotype_A].index.astype(int)
    )
    b_ix = list(
        adata.obs[adata.obs[phenotype_column] == phenotype_B].index.astype(int)
    )
    ab = sym[np.ix_(a_ix, b_ix)]  # A rows -> B cols
    ba = sym[np.ix_(b_ix, a_ix)]  # B rows -> A cols

    total_cells = sum(ab.shape)

    # Count the number of nodes of pair A and B that neighbor each other / totals
    if method == "nodes":
        if isinstance(adjacency_matrix, np.ndarray):
            f_sum = ab.any(
                axis=1
            ).sum()  # How many A cells have atleast 1 B neighbor
            s_sum = ba.any(
                axis=1
            ).sum()  # How many B cells have atleast 1 A neighbor

        elif isinstance(adjacency_matrix, csr_matrix):
            f_sum = (ab.getnnz(axis=1) > 0).sum()
            s_sum = (ba.getnnz(axis=1) > 0).sum()

        else:
            raise ValueError("invalid adjacency matrix type")

        total_interactions = (
            f_sum + s_sum
        )  # Represents total number of interacting cells in A and B

    # Count the number of times pair A and B neighbor each other / totals
    elif method == "edges":
        f_sum = (
            ab.sum()
        )  # How many B neighbors every A cells have in the graph
        s_sum = (
            ba.sum()
        )  # How many A neighbors every B cells have in the graph

        total_interactions = (
            f_sum + s_sum
        )  # Represents total number of interactions between A and B

    else:
        raise ValueError("invalid method")

    # Account for self comparisons. Normalised by density, but need to report counts
    if phenotype_A == phenotype_B:
        total_interactions = total_interactions / 2
        total_cells = total_cells / 2

    # # Minimum number of cells for a comparison
    not_enough_cells = total_cells < 2
    # For different phenotypes. If self, then not_enough_cells will be 0 anyway
    not_enough_of_category = len(a_ix) == 0 or len(b_ix) == 0

    missing = False
    if not_enough_cells or not_enough_of_category:
        missing = True

    return total_interactions, total_cells, missing


def proximity_density(
    adata: AnnData,
    grouping: str,
    phenotype: str,
    pairs: list[tuple[str, str]] = None,
    connectivity_key: str = "spatial_connectivities",
    multi_index: bool = False,
    inplace: bool = True,
    n_jobs: int = 4,
) -> None | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Computes proximity density from scimap's p-score function compatible with
    squidpy-generated .obsp spatial graphs. By default, stores the results
    inplace.

    Proximity density is defined as the number of cells of a given pair of
    phenotypes being in proximity of one another, divided by the total number of
    cells.

    The definition of proximity depends on the adjacency matrix computed via
    squidpy.gr.spatial_neighbors. To stay true to the original definition of
    proximity density, the adjacency matrix should be a radial graph of a given
    radius in um.

    Args:
        adata: AnnData object.
        grouping: Column name in adata.obs to group by.
        phenotype: Column name in adata.obs to compute proximity for.
        pairs: List of tuples of phenotype pairs to compute proximity for.
            If None, computes proximity for all unique phenotype pairs.
        connectivity_key: Key for the adjacency matrix in adata.obsp.
        multi_index: If True, returns a multi-indexed DataFrame.
        inplace: If True, stores the results in adata.uns.

    Returns:
        If inplace is False, returns a tuple of three dataframes, the
        first containing the proximity density results, the second containing
        the masks for missing values, and the third containing the cell counts
        for each pair of phenotypes.
    """
    # Drop na phenotype rows
    adata = adata[~adata.obs[phenotype].isna()]

    if connectivity_key not in adata.obsp:
        raise ValueError("No adjacency matrix found in adata.obsp.")

    if grouping not in adata.obs.columns:
        raise ValueError("Grouping column not found in adata.obs.")

    if phenotype not in adata.obs.columns:
        raise ValueError("Phenotype column not found in adata.obs.")

    if pairs is None:
        phenotypes = list(adata.obs[phenotype].unique())
        pairs = list(combinations_with_replacement(phenotypes, 2))

    labels = (phenotype, f"neighbour_{phenotype}")

    adata_list = [
        adata[adata.obs[grouping] == g] for g in adata.obs[grouping].unique()
    ]

    def _process_adata_subset(
        adata_subset, pairs, phenotype, connectivity_key, grouping
    ):
        group = adata_subset.obs[grouping].unique()[0]

        densities = {}
        masks = {}
        counts = {}

        for pair in pairs:
            total_interactions, total_cells, missing = (
                compute_pair_interactions(
                    adata=adata_subset,
                    phenotype_column=phenotype,
                    phenotype_A=pair[0],
                    phenotype_B=pair[1],
                    method="nodes",
                    connectivity_key=connectivity_key,
                )
            )

            if total_cells == 0:
                densities[pair] = 0
            else:
                densities[pair] = total_interactions / total_cells
            masks[pair] = missing
            counts[pair] = total_cells

        return group, densities, masks, counts

    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_adata_subset)(
            adata_subset, pairs, phenotype, connectivity_key, grouping
        )
        for adata_subset in adata_list
    )

    grouping_comparisons = {}
    mask_comparisons = {}
    count_comparisons = {}

    for group, densities, masks, counts in results:
        grouping_comparisons[group] = densities
        mask_comparisons[group] = masks
        count_comparisons[group] = counts

    grouping_df = pd.DataFrame(grouping_comparisons)
    grouping_df.index = grouping_df.index.set_names(labels)
    grouping_df.columns.name = grouping
    mask_df = pd.DataFrame(mask_comparisons)
    mask_df.index = mask_df.index.set_names(labels)
    mask_df.columns.name = grouping
    count_df = pd.DataFrame(count_comparisons)
    count_df.index = count_df.index.set_names(labels)
    count_df.columns.name = grouping

    if not multi_index:
        grouping_df = grouping_df.reset_index()
        mask_df = mask_df.reset_index()
        count_df = count_df.reset_index()

    if inplace:
        adata.uns["proximity_density_results"] = grouping_df
        adata.uns["proximity_density_masks"] = mask_df
        adata.uns["proximity_density_cell_counts"] = count_df
        return adata
    else:
        return grouping_df, mask_df, count_df

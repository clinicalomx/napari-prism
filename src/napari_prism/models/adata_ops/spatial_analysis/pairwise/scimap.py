"""Spatial analyses which generate metrics at the cell level."""

from itertools import combinations_with_replacement
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from joblib import Parallel, delayed

from napari_prism.models.adata_ops.spatial_analysis.graph import (
    compute_pair_interactions,
)
from napari_prism.models.adata_ops.spatial_analysis.schema import CellEntity


@xr.register_dataarray_accessor("proximity_density")
class ProximityDensityMetricAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def validate(self):
        """
        a) Ensure there's a second population to compare against.
        b) Enforce exclusively global vs global or compartment vs compartment
            comparisons.
        """
        self._obj.spatial_metric.validate()
        assert "radius" in self._obj.coords
        cell_population_a = self._obj.coords.get("cell_population_a", None)
        cell_population_b = self._obj.coords.get("cell_population_b", None)

        if cell_population_b is None:
            raise AttributeError(
                "Need a second cell population for GCross metrics."
            )

        def _retrieve_compartment(label):
            if "@" in label:
                return label.split("@")[-1]
            else:
                return None

        compartment_a = _retrieve_compartment(cell_population_a)
        compartment_b = _retrieve_compartment(cell_population_b)

        if (compartment_a is None) != (compartment_b is None):
            raise ValueError(
                "Cannot compare global populations to compartmental"
                "populations"
            )
        return self._obj

    def plot(self):
        self._obj.spatial_metric.plot()

    def pretty_print(self):
        self._obj.spatial_metric.pretty_print()

    def query(self):
        pass

    def test(self, adata):
        """Test if GCross curve is beyond some homogenous process."""
        indices = self._obj.sample_id  # noqa: F841
        # convex hull;
        # estimate lambda / density
        # theoretical G under given Xs / radii
        # Test observed G vs theoretical G


def create_proximity_density_metric(
    values: np.ndarray,
    cell_population_a: CellEntity,
    cell_population_b: CellEntity,
    sample_id: str,
    parameters: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Create a proximity density result DataArray.

    Args:

    """


def proximity_density(
    adata: AnnData,
    grouping: str,
    phenotype: str,
    pairs: list[tuple[str, str]] | None = None,
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

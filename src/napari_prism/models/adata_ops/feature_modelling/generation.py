# Brute force all combinations of cell types x compartmentsde
import anndata as ad
import pandas as pd
import xarray as xr

from napari_prism.models.adata_ops.feature_modelling.obs import (
    cell_proportions,
)
from napari_prism.models.adata_ops.spatial_analysis.pairwise.gcross import (
    gcross,
)
from napari_prism.models.adata_ops.spatial_analysis.pairwise.jsd import jsd


def generate_all_spatial_features(
    adata: ad.AnnData,
    sample_key: str,
    cell_type_key: str,
    spatial_key: str = "spatial",
    gcross_kwargs: dict | None = None,
    jsd_kwargs: dict | None = None,
) -> xr.Dataset:
    """Generates all pairwise spatial features. Excludes barrier_score.

    Turns Gcross curve into an AUC summary.
    """

    # parse gcross_kwargs for signature matching parameters
    if gcross_kwargs:
        gcross_scores = gcross(
            adata=adata,
            sample_key=sample_key,
            cell_type_key=cell_type_key,
            spatial_key=spatial_key,
            **gcross_kwargs,
        )
    else:
        gcross_scores = gcross(
            adata=adata,
            sample_key=sample_key,
            cell_type_key=cell_type_key,
            spatial_key=spatial_key,
        )
    gcross_scores = gcross_scores.gcross.auc()

    # parse jsd scores
    if jsd_kwargs:
        jsd_scores = jsd(
            adata=adata,
            sample_key=sample_key,
            cell_type_key=cell_type_key,
            spatial_key=spatial_key,
            **jsd_kwargs,
        )
    else:
        jsd_scores = jsd(
            adata=adata,
            sample_key=sample_key,
            cell_type_key=cell_type_key,
            spatial_key=spatial_key,
        )

    # All
    spatial_scores = xr.combine_by_coords([gcross_scores, jsd_scores])
    return spatial_scores


def generate_all_proportional_features(
    adata: ad.AnnData,
    sample_key: str,
    cell_type_key: str,
    compartment_key: str | None = None,
    # misc_key: str | list[str] | None = None,  # other metadatas
    normalise_by_compartment: bool = False,
) -> xr.DataArray:
    proportions = cell_proportions(
        adata=adata,
        sample_key=sample_key,
        obs_column=cell_type_key,
        compartment_column=compartment_key,
        normalise_by_compartment=normalise_by_compartment,
    )

    return proportions


def flatten_metric(da: xr.DataArray, sample_dim="sample_id") -> xr.DataArray:
    """
    Flatten any DataArray into shape (sample_id x feature)
    while keeping metadata in a MultiIndex.
    """
    # Identify non-sample dimensions
    feature_dims = [d for d in da.dims if d != sample_dim]

    if len(feature_dims) == 0:
        # Already 1D per sample
        da_flat = da.rename({da.dims[0]: "feature"})
        da_flat = da_flat.expand_dims("feature")
        da_flat = da_flat.assign_coords(feature=[da.name or "value"])
        return da_flat.transpose(sample_dim, "feature")

    # Stack all non-sample dims into 'feature'
    da_flat = da.stack(feature=feature_dims)

    # Build MultiIndex for metadata
    feature_index = pd.MultiIndex.from_arrays(
        [da_flat[dim].values for dim in feature_dims], names=feature_dims
    )
    da_flat = da_flat.assign_coords(feature=feature_index)

    # Transpose to (sample_dim x feature)
    da_flat = da_flat.transpose(sample_dim, "feature")
    return da_flat

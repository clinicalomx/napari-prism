# Brute force all combinations of cell types x compartmentsde
import anndata as ad
import numpy as np
import xarray as xr

from napari_prism.models.adata_ops.feature_modelling.obs import (
    cell_proportions,
    get_sample_covariate,
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
    misc_key: str | list[str] | None = None,  # other metadatas
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


def add_covariate_to_dataset(
    dataset: xr.Dataset,
    adata: ad.AnnData,
    sample_key: str,
    covariate_key: str,
) -> xr.Dataset:
    """Add covariate from adata.obs to dataset as a variable.

    Covariates are added as variables commonly indexed by sample_key to the
    dataset, and can be used for stratification or adjustment in downstream
    analyses.

    Args:
        dataset: xr.Dataset with sample_id dimension.
        adata: AnnData with obs containing covariates.
        sample_key: Key in adata.obs that matches dataset sample_id coord.
        covariate_key: Key or list of keys in adata.obs to add as covariates.

    Returns:
        xr.Dataset with added covariate coords.
    """
    original_dim_order = tuple(dataset.dims.keys())
    ys = get_sample_covariate(adata, sample_key, covariate_key)
    ys.index.name = "sample_id"
    dataset = dataset.copy()
    dataset[covariate_key] = ys.to_xarray().set_coords(covariate_key)[
        covariate_key
    ]
    dataset[covariate_key] = dataset[covariate_key].astype(str)
    dataset = dataset.transpose(*original_dim_order, ...)
    return dataset


def add_survival_covariate_to_dataset(
    dataset: xr.Dataset,
    adata: ad.AnnData,
    sample_key: str,
    event_key: str,
    time_key: str,
    survival_key_added: str = "Survival",
) -> xr.Dataset:
    """Add survival covariate from adata.obs to dataset as a variable.

    Survival covariates are added as variables commonly indexed by sample_key
    to the dataset, and can be used for stratification or adjustment in
    downstream analyses.

    Args:
        dataset: xr.Dataset with sample_id dimension.
        adata: AnnData with obs containing covariates.
        sample_key: Key in adata.obs that matches dataset sample_id coord.
        event_key: Key in adata.obs for event (boolean).
        time_key: Key in adata.obs for time (numeric).

    Returns:
        xr.Dataset with added survival covariate coords.
    """
    # check if it has an original key
    if sample_key not in adata.obs and (
        "original_key" in dataset["sample_id"].attrs
    ):
        sample_key = dataset["sample_id"].attrs["original_key"]
    os = get_sample_covariate(adata, sample_key, [event_key, time_key])
    os.index.name = "sample_id"
    os_data = np.stack([os[time_key].values, os[event_key].values], axis=1)
    dataset = dataset.copy()
    dataset[survival_key_added] = xr.DataArray(
        os_data,
        coords={"sample_id": os.index, "survival": ["time", "event"]},
        dims=["sample_id", "survival"],
    )
    return dataset

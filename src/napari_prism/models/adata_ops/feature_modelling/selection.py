"""Recipes for running the feature selection pipelines."""

import anndata as ad
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display_html
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import validate_data
from sksurv.util import Surv
from stabl.preprocessing import LowInfoFilter
from stabl.stabl import Stabl

from napari_prism.models.adata_ops.feature_modelling.generation import (
    add_survival_covariate_to_dataset,
)
from napari_prism.models.adata_ops.feature_modelling.survival import (
    CoxnetFitter,
)


class PatchedLowInfoFilter(LowInfoFilter):  # For sklearn 1.7.0
    def _validate_data(self, *args, **kwargs):
        return validate_data(self, *args, **kwargs)


class PatchedStabl(Stabl):
    def _validate_data(self, *args, **kwargs):
        return validate_data(self, *args, **kwargs)


def flatten_metric_flatcoords(
    da: xr.DataArray, sample_dim="sample_id"
) -> xr.DataArray:
    """
    Flatten any DataArray into shape (sample_id x feature)
    while keeping metadata encoded in string feature names,
    instead of a MultiIndex (for sklearn compatibility).
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

    # Build flat string labels
    labels = [
        "|".join(str(x) for x in vals)
        for vals in zip(
            *[da_flat[dim].values for dim in feature_dims], strict=False
        )
    ]

    da_flat = da_flat.assign_coords(feature=labels)

    # Drop the original index dims (since we encoded them in labels)
    da_flat = da_flat.drop_vars(feature_dims, errors="ignore")

    # Transpose to (sample_dim x feature)
    da_flat = da_flat.transpose(sample_dim, "feature")
    return da_flat


def unflatten_from_flatcoords(
    da_flat: xr.DataArray, original_dims: list[str], sample_dim="sample_id"
) -> xr.DataArray:
    """
    Reconstruct multi-dimensional DataArray from flat string feature names.

    Parameters:
        da_flat : xr.DataArray
            Flattened array with 'feature' strings
        original_dims : list[str]
            List of original feature dimensions (excluding sample_dim)
        sample_dim : str
            Name of sample dimension

    Returns:
        xr.DataArray
    """
    # Split string labels into lists per feature
    split_vals = [label.split("|") for label in da_flat["feature"].values]

    # Convert to MultiIndex
    multi_idx = pd.MultiIndex.from_tuples(split_vals, names=original_dims)

    # Assign as new feature index
    global da_unstacked
    da_flat = da_flat.assign_coords(feature=multi_idx)

    # Unstack into original dims
    da_unstacked = da_flat.unstack("feature")

    return da_unstacked


def feature_indices_to_locs(feature_names_out):
    locs = [int(f.lstrip("x")) for f in feature_names_out]
    return locs


def traverse_pipeline_graph(pipeline):
    prev_xilocs = None
    for step_ix, p in enumerate(pipeline):
        xilocs = p.get_feature_names_out()
        xilocs = np.array(feature_indices_to_locs(xilocs))
        if prev_xilocs is not None and step_ix > 0:
            xilocs = prev_xilocs[xilocs]
        prev_xilocs = xilocs
    return prev_xilocs


def default_processing_pipeline():
    processing_steps = [
        ("variance_threshold", VarianceThreshold(threshold=0.001)),
        (
            "low_info_filter",
            PatchedLowInfoFilter(),
        ),  # if feat has 40% NaNs, remove it
        (
            "median_imputer",
            SimpleImputer(),
        ),  # For impute as median of the feature
        ("zscore", StandardScaler()),
    ]

    processing_pipeline = Pipeline(processing_steps)
    return processing_pipeline


def stabl_survival_pipeline(
    feature_dataset: xr.Dataset,
    adata: ad.AnnData,
    event_key: str,
    time_key: str,
    base_estimator: BaseEstimator | None = None,
    survival_key_added: str = "Survival",
    sample_key: str = "sample_id",
    processing_pipeline: Pipeline | None = None,
    train_size: float = 0.8,
    n_bootstraps: int = 200,
    alpha_grid: list[float] | None = None,
):
    def patch_attributes(stabl_obj: PatchedStabl):
        d = stabl_obj.fitted_lambda_grid_
        d["alpha"] = d.pop("alphas")

    assert train_size < 1.0

    flat_data = flatten_metric_flatcoords(
        feature_dataset, sample_dim="sample_id"
    )

    flat_data = add_survival_covariate_to_dataset(
        flat_data,
        adata,
        sample_key=sample_key,
        event_key=event_key,
        time_key=time_key,
        survival_key_added=survival_key_added,
    )

    # Drop features that are completely NaN
    flat_data = flat_data.dropna(dim="feature")
    # Drop samples that are NAN survival data (censorship should be 0 not NaN)
    flat_data = flat_data.dropna(dim="survival")

    if processing_pipeline is None:
        processing_pipeline = default_processing_pipeline()

    if base_estimator is None:
        base_estimator = CoxnetFitter()

    if alpha_grid is None:
        alpha_grid = np.geomspace(0.08, 1, 15)

    stabl = PatchedStabl(
        base_estimator=base_estimator,
        lambda_grid={
            "alphas": [[x] for x in alpha_grid],
            "l1_ratio": [0.9],
        },
        n_bootstraps=n_bootstraps,
        artificial_type="knockoff",  # or "random_permutation"
        artificial_proportion=1.0,  # 1:1 spike ins
        sample_fraction=0.65,  # subsampling fraction
        random_state=42,
        verbose=30,
        n_jobs=30,
    )

    stabl_survival_pipeline = Pipeline(
        steps=processing_pipeline.steps + [("stabl_survival", stabl)],
        verbose=True,
    )

    # if display
    X = flat_data["features"]
    y = flat_data[survival_key_added]

    # Train test split by indices
    train_indices = np.random.choice(
        np.arange(len(X.sample_id)),
        size=int(len(X.sample_id) * train_size),
        replace=False,
    )
    test_indices = np.setdiff1d(np.arange(len(X.sample_id)), train_indices)

    X_train = X.isel(sample_id=train_indices)
    X_test = X.isel(sample_id=test_indices)
    y_test = y.isel(sample_id=test_indices)
    y_train = y.isel(sample_id=train_indices)

    y_test = Surv().from_arrays(
        event=y_test.sel(survival="event").values.astype(bool),
        time=y_test.sel(survival="time").values.astype(float),
    )

    y_train = Surv().from_arrays(
        event=y_train.sel(survival="event").values.astype(bool),
        time=y_train.sel(survival="time").values.astype(float),
    )

    # Execute;
    display_html(stabl_survival_pipeline)
    results = stabl_survival_pipeline.fit(X_train, y_train)
    patch_attributes(results.named_steps["stabl_survival"])

    final_feature_indices = traverse_pipeline_graph(results)

    # Re-fit on data

    # Heuristic;
    # if len(final_feature_indices) > 30:
    # apply some regularization, otherwise just do standard coxnet
    final_estimator = clone(base_estimator)
    if len(final_feature_indices) > 30:
        final_l1_ratio = 1.0
        final_alphas = 0.1
    else:
        final_l1_ratio = 1.0
        final_alphas = 0.0
    final_estimator.set_params(l1_ratio=final_l1_ratio, alphas=[final_alphas])

    final_estimator.fit(X_train.isel(feature=final_feature_indices), y_train)

    return final_estimator, X_test, y_test, final_feature_indices, results


def stabl_binary():
    pass
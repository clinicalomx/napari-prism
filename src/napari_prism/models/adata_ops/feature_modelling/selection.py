"""Recipes for running the feature selection pipelines."""

import xarray as xr
from sklearn.utils.validation import validate_data
from stabl.preprocessing import LowInfoFilter


def stabl_survival():
    pass


def stabl_binary():
    pass


# data = xr.combine_by_coords([p, s])


# X = load_digits_dataarray()
# y = Target(coord="digit")(X)


class PatchedLowInfoFilter(LowInfoFilter):  # For sklearn 1.7.0
    def _validate_data(self, *args, **kwargs):
        return validate_data(self, *args, **kwargs)


# processing_steps = [
#     ("variance_threshold", wrap(VarianceThreshold(threshold=0), sample_dim="sample_id", reshapes="feature")),
#     ("low_info_filter", wrap(PatchedLowInfoFilter(max_nan_fraction=0.4), sample_dim="sample_id", reshapes="feature")), # if feat has 40% NaNs, remove it
#     ("median_imputer", wrap(SimpleImputer(strategy="median"), sample_dim="sample_id", reshapes="feature")),  # For remaining Nans, impute as median of the feature
#     # ("knn_imputer", KNNImputer()) -> TODO: based on similar feature families, i.e. ct comparisons
#     ("zscore", wrap(StandardScaler(), sample_dim="sample_id", reshapes="feature"))
# ]
# processing_pipeline = Pipeline(
#     processing_steps,
#     verbose=True
# )

# flat = flatten_to_2d(data)
# flat
# processing_pipeline.fit_transform(flat)


def flatten_to_2d(ds: xr.Dataset, sample_dim="sample_id") -> xr.DataArray:
    """
    Flatten an xarray.Dataset (with multiple variables and feature dims)
    into a 2D DataArray of shape (sample, feature).
    """
    # Convert Dataset → DataArray with variable as an extra dim
    da = ds.to_array("variable")

    # Collapse everything except sample_dim into a single "feature" dim
    stacked = da.stack(feature=[d for d in da.dims if d != sample_dim])

    # Return a simple 2D DataArray
    return stacked.transpose(sample_dim, "feature")


def unflatten_from_2d(
    flat: xr.DataArray, template: xr.Dataset, sample_dim="sample_id"
) -> xr.DataArray:
    """
    Reverse the flattening, restoring original dims from the template Dataset.
    """
    da = template.to_array("variable").stack(
        feature=[d for d in template.dims if d != sample_dim]
    )
    coords = {sample_dim: template[sample_dim], "feature": da.feature}

    restored = xr.DataArray(
        flat.values,
        dims=[sample_dim, "feature"],
        coords=coords,
        attrs=template.attrs,
    )
    return restored.unstack("feature")


# X = load_dummy_dataarray()

# scaler = wrap(StandardScaler(), sample_dim="sample_id", reshapes="features")
# lowinfo = wrap(PatchedLowInfoFilter(), sample_dim="sample_id", reshapes="features")
# flat = flatten_to_2d(data, sample_dim="sample_id")
# Xl = lowinfo.fit_transform(flat)
# Xt = scaler.fit_transform(flat)
# Xt
# df = data.to_dataframe()
# df.reset_index().pivot_table(
#     index="sample_id",
#     columns=["metric_name", "cell_population_a", "cell_population_b"],
#     values=df.columns.difference(["sample_id", "metric_name", "cell_population_a", "cell_population_b"])
# )

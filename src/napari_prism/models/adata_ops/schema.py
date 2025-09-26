"""
Dev Notes;
- For xarray integration, should consider composition over inheritance.
- Spatial metrics can be sought to be a multiple inherit. problem;
    - Can be pairwise, self or not
    - Can be compartmental or not
    - Can have xs, ys or not
    etc
- Added accessors for spatial metric function modules
- Place entity classes in higher level module for standardized string reprs
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData


@dataclass(frozen=True)
class CompartmentEntity:
    """Represents a compartment in the tissue."""

    compartment: str
    compartment_col: str

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.compartment_col}={self.compartment}"

    def __eq__(self, value):
        return self.compartment == value.compartment_name

    def to_xr_variable(self, var_name):
        attrs = {"compartment_column": self.compartment_col}
        return xr.Variable(
            (var_name,),
            [self.compartment],
            attrs=attrs,
        )


@dataclass(frozen=True)
class CellEntity:
    """Represents a population of a particular cell type"""

    cell_type: str
    cell_type_col: str

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.cell_type_col}={self.cell_type}"

    def to_xr_variable(self, var_name):
        attrs = {"cell_type_column": self.cell_type_col}
        return xr.Variable(
            (var_name,),
            [self.cell_type],
            attrs=attrs,
        )


@dataclass
class SampleEntity:
    """Represents samples from which to perform comparative analysis."""

    sample_id: str
    features: list[xr.DataArray] | xr.Dataset

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if isinstance(self.features, list):
            dsc = f"sample_id: {self.sample_id}, {len(self.features)} features"
            return dsc
        elif isinstance(self.features, xr.Dataset):
            pass
        else:
            pass

    def to_adata(self):
        pass


# calling dataset.spatial_metric.method
@xr.register_dataarray_accessor("metric")
class MetricAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def validate(self):
        """Validate Metric schema"""
        required_coords = {
            "sample_id",
            "metric_name",
            "cell_population_a",
            "cell_compartment_a",
            "cell_population_b",
            "cell_compartment_b",
        }
        if not required_coords.issubset(self._obj.coords):
            raise ValueError(
                f"Missing required coordinates: "
                f"{required_coords - set(self._obj.coords)}"
            )
        return self._obj

    def plot(self):
        """Placeholder for specific metrics"""
        return

    def pretty_print(self):
        """Nicer representation"""
        coords = self._obj.coords
        metric_name = coords.get("metric_name", None)
        print(metric_name)
        cell_population_a = coords.get("cell_population_a", None)
        cell_population_b = coords.get("cell_population_b", None)
        directional = coords.get("directional", None)

        if cell_population_b:
            if directional:
                pair = f"{cell_population_a} -> {cell_population_b}"
            else:
                pair = f"{cell_population_a} <-> {cell_population_b}"
        else:
            pair = f"{cell_population_a}"
        return f"SpatialMetric {metric_name} " f"[{pair}]"


def create_metric(
    values: np.ndarray,
    sample_id: str,
    cell_population_a: CellEntity,
    cell_compartment_a: CompartmentEntity | None = None,
    sample_id_key: str | None = None,
    dims: list[str] | None = None,
    coords: dict[str, Any] | None = None,
    cell_population_b: CellEntity | None = None,
    cell_compartment_b: CompartmentEntity | None = None,
    metric_name: str = "generic",
    directional: bool = False,
    parameters: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Create a Metric as a DataArray with common metadata.
    The DataArray will always have sample_id and metric_name as the first two
    dimensions, followed by any other dimensions specified in dims.

    Parameters for this metric are stored as a coordinate aligned with metric_name.
    """
    import json

    static_dims = ["sample_id", "metric_name", "cell_population_a"]
    all_coords = {}

    # sample_id is length-1 along its dim
    all_coords["sample_id"] = ("sample_id", [sample_id])

    # cell_population_a
    all_coords["cell_population_a"] = cell_population_a.to_xr_variable(
        "cell_population_a"
    )

    # cell_compartment_a
    if cell_compartment_a is None:
        cell_compartment_a = CompartmentEntity("None", "None")
    all_coords["cell_compartment_a"] = cell_compartment_a.to_xr_variable(
        "cell_compartment_a"
    )
    static_dims += ["cell_compartment_a"]

    # cell_population_b
    if cell_population_b is None:
        cell_population_b = CellEntity("None", "None")
    all_coords["cell_population_b"] = cell_population_b.to_xr_variable(
        "cell_population_b"
    )
    static_dims += ["cell_population_b"]

    # cell_compartment_b
    if cell_compartment_b is None:
        cell_compartment_b = CompartmentEntity("None", "None")
    all_coords["cell_compartment_b"] = cell_compartment_b.to_xr_variable(
        "cell_compartment_b"
    )
    static_dims += ["cell_compartment_b"]

    # Extra dims
    all_dims = static_dims + (dims if dims else [])
    if coords:
        for k, v in coords.items():
            if (
                isinstance(v, tuple)
                and len(v) == 2
                and isinstance(v[0], list | tuple)
            ):
                all_coords[k] = v
            else:
                all_coords[k] = (k, v)

    # Prepare data array shape
    arr = np.asarray(values)
    target_ndim = len(all_dims)
    if arr.ndim > target_ndim:
        raise ValueError(
            f"values has too many dimensions: values.ndim={arr.ndim} > target dims={target_ndim}"
        )
    arr = arr.reshape((1,) * (target_ndim - arr.ndim) + arr.shape)

    # Create DataArray
    da = xr.DataArray(data=arr, dims=all_dims, coords=all_coords)

    if sample_id_key is not None:
        da["sample_id"].attrs["original_key"] = sample_id_key

    # Store parameters as a coordinate aligned with metric_name
    full_params = dict(parameters or {})
    full_params["directional"] = (
        directional  # include directional in parameters
    )
    # da = da.expand_dims(metric_name=[metric_name])

    if "metric_name" not in da.dims:
        da = da.expand_dims({"metric_name": [metric_name]})

    da = da.assign_coords(
        metric_name=("metric_name", [metric_name]),
    )

    da.coords["metric_name"].attrs[f"{metric_name}_parameters"] = json.dumps(
        full_params
    )

    da = da.metric.validate()
    return da


def xarray_to_anndata(
    data: xr.DataArray | xr.Dataset, sample_id_dim: str = "sample_id"
) -> AnnData:
    """
    Convert structured xarray metric data to AnnData format.

    Args:
        data : xr.DataArray or xr.Dataset
            Structured metric data with dimensions including sample_id and
            feature metadata
        sample_id_dim : str, default "sample_id"
            Name of the dimension containing sample identifiers

    Returns:
        AnnData object with:
        - .obs: DataFrame indexed by sample_id
        - .X: Metric values as dense array
        - .var: Feature metadata (cell populations, compartments, etc.)
        - .uns: Metric parameters and metadata
    """
    if isinstance(data, xr.Dataset):
        # If Dataset, concatenate all DataArrays along a new feature dimension
        data_arrays = []
        for var_name, da in data.data_vars.items():
            # Add variable name as a coordinate
            da_with_name = da.assign_coords(feature_name=var_name)
            data_arrays.append(da_with_name)
        data = xr.concat(data_arrays, dim="feature")

    # Ensure sample_id_dim exists
    if sample_id_dim not in data.dims:
        raise ValueError(f"Dimension '{sample_id_dim}' not found in data")

    # Get sample IDs for obs index
    sample_ids = data.coords[sample_id_dim].values

    # Create feature names by combining metadata coordinates
    feature_coords = [
        coord
        for coord in data.coords
        if coord != sample_id_dim and coord in data.dims
    ]

    # Build feature metadata dataframe
    var_data = {}
    feature_names = []

    # Handle multi-dimensional features
    if len(feature_coords) > 1:
        # Create cartesian product of all feature coordinates
        coord_values = [data.coords[coord].values for coord in feature_coords]
        coord_combinations = np.array(np.meshgrid(*coord_values)).T.reshape(
            -1, len(feature_coords)
        )

        for _, combo in enumerate(coord_combinations):
            feature_parts = []
            for j, coord_name in enumerate(feature_coords):
                coord_val = combo[j]
                feature_parts.append(f"{coord_name}={coord_val}")
            feature_names.append("|".join(feature_parts))

            # Store individual coordinate values in var
            for j, coord_name in enumerate(feature_coords):
                if coord_name not in var_data:
                    var_data[coord_name] = []
                var_data[coord_name].append(combo[j])
    elif len(feature_coords) == 1:
        coord_name = feature_coords[0]
        coord_values = data.coords[coord_name].values
        feature_names = [f"{coord_name}={val}" for val in coord_values]
        var_data[coord_name] = coord_values
    else:
        # Single feature case
        feature_names = ["metric_value"]

    # Extract metric values and reshape for AnnData (.X should be samples x features)
    values = data.values

    # Move sample_id dimension to first axis if needed
    sample_axis = data.get_axis_num(sample_id_dim)
    if sample_axis != 0:
        values = np.moveaxis(values, sample_axis, 0)

    # Flatten all non-sample dimensions into features
    n_samples = values.shape[0]
    values_2d = values.reshape(n_samples, -1)

    # Ensure we have the right number of features
    if len(feature_names) != values_2d.shape[1]:
        # Generate generic feature names if mismatch
        feature_names = [f"feature_{i}" for i in range(values_2d.shape[1])]
        var_data = {"feature_index": list(range(values_2d.shape[1]))}

    # Create var DataFrame with feature metadata
    var_df = (
        pd.DataFrame(var_data, index=feature_names)
        if var_data
        else pd.DataFrame(index=feature_names)
    )

    # # Extract coordinate attributes for var metadata
    # for coord_name in data.coords:
    #     if hasattr(data.coords[coord_name], 'attrs') and data.coords[coord_name].attrs:
    #         for attr_key, attr_val in data.coords[coord_name].attrs.items():
    #             var_df[f"{coord_name}|{attr_key}"] = attr_val

    # Create obs DataFrame (sample metadata)
    obs_df = pd.DataFrame(index=sample_ids)

    # Extract sample-level metadata from coordinates
    for coord_name, coord_obj in data.coords.items():
        if (
            (coord_name != sample_id_dim)
            and (coord_name not in feature_coords)
            and (coord_obj.dims == (sample_id_dim,))
        ):
            obs_df[coord_name] = coord_obj.values

    # Create uns dictionary with metric parameters and global metadata
    uns_data = {}

    # Store data attributes
    if hasattr(data, "attrs") and data.attrs:
        uns_data.update(data.attrs)

    # Store coordinate attributes that aren't feature-specific
    for coord_name, coord_obj in data.coords.items():
        if hasattr(coord_obj, "attrs") and coord_obj.attrs:
            for attr_key, attr_val in coord_obj.attrs.items():
                uns_data[f"{coord_name}|{attr_key}"] = attr_val

    # Store original dimension information for reconstruction
    uns_data["_xarray_dims"] = list(data.dims)
    uns_data["_xarray_coords"] = list(data.coords.keys())
    uns_data["_sample_id_dim"] = sample_id_dim

    # Create AnnData object
    adata = AnnData(X=values_2d, obs=obs_df, var=var_df, uns=uns_data)

    return adata


def anndata_to_xarray(
    adata: AnnData,
    sample_id_dim: str | None = None,
    reconstruct_dims: bool = True,
) -> xr.DataArray:
    """
    Convert AnnData back to structured xarray format.

    Parameters
    ----------
    adata : AnnData
        AnnData object to convert
    sample_id_dim : str, optional
        Name for the sample dimension. If None, uses stored value from .uns
    reconstruct_dims : bool, default True
        Whether to attempt to reconstruct original xarray dimensions

    Returns
    -------
    xr.DataArray
        Structured metric data with original coordinate structure
    """
    # Get sample_id dimension name
    if sample_id_dim is None:
        sample_id_dim = adata.uns.get("_sample_id_dim", "sample_id")

    # Extract data matrix
    X = adata.X
    if hasattr(X, "toarray"):  # Handle sparse matrices
        X = X.toarray()

    # Get sample IDs and feature names
    sample_ids = adata.obs.index.values
    feature_names = adata.var.index.values

    # Initialize coordinates
    coords = {sample_id_dim: sample_ids}

    # Add sample-level coordinates from obs
    for col in adata.obs.columns:
        coords[col] = (sample_id_dim, adata.obs[col].values)

    if reconstruct_dims and "_xarray_dims" in adata.uns:
        # Attempt to reconstruct original structure
        original_dims = adata.uns["_xarray_dims"]
        original_coords = adata.uns.get("_xarray_coords", [])

        # Parse feature names to extract coordinate values
        feature_coords = {}
        for coord_name in original_coords:
            if (
                coord_name != sample_id_dim
                and coord_name not in adata.obs.columns
            ):
                if coord_name in adata.var.columns:
                    feature_coords[coord_name] = adata.var[coord_name].unique()
                else:
                    # Try to parse from feature names
                    coord_values = []
                    for feature_name in feature_names:
                        if f"{coord_name}=" in feature_name:
                            parts = feature_name.split("|")
                            for part in parts:
                                if part.startswith(f"{coord_name}="):
                                    val = part.split("=", 1)[1]
                                    if val not in coord_values:
                                        coord_values.append(val)
                    if coord_values:
                        feature_coords[coord_name] = coord_values

        # Add feature coordinates
        coords.update(feature_coords)

        # Reconstruct data shape
        try:
            shape = [len(coords[dim]) for dim in original_dims]
            reshaped_data = X.reshape(shape)
            dims = original_dims
        except (ValueError, KeyError):
            # Fall back to simple 2D structure
            dims = [sample_id_dim, "feature"]
            coords["feature"] = feature_names
            reshaped_data = X
    else:
        # Simple 2D structure
        dims = [sample_id_dim, "feature"]
        coords["feature"] = feature_names
        reshaped_data = X

    # Create DataArray
    da = xr.DataArray(data=reshaped_data, dims=dims, coords=coords)

    # Restore coordinate attributes from var metadata
    for col in adata.var.columns:
        if "|" in col:
            coord_name, attr_name = col.split("|", 1)
            if coord_name in da.coords:
                if not hasattr(da.coords[coord_name], "attrs"):
                    da.coords[coord_name].attrs = {}
                # Get first non-null value as the attribute
                attr_val = (
                    adata.var[col].dropna().iloc[0]
                    if not adata.var[col].dropna().empty
                    else None
                )
                if attr_val is not None:
                    da.coords[coord_name].attrs[attr_name] = attr_val

    # Restore global attributes from uns
    attrs = {}
    for key, val in adata.uns.items():
        if not key.startswith("_"):  # Skip internal metadata
            if "|" in key:
                coord_name, attr_name = key.split("|", 1)
                if coord_name in da.coords:
                    if not hasattr(da.coords[coord_name], "attrs"):
                        da.coords[coord_name].attrs = {}
                    da.coords[coord_name].attrs[attr_name] = val
                else:
                    attrs[key] = val
            else:
                attrs[key] = val

    da.attrs.update(attrs)

    return da


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


def grow_feature_hypercube(dataarrays, keep_order="first"):
    """
    Util function to combine iteratively grow the DataArray hypercubes. Useful
    for growing the feature database dynamically. If new feature introduces
    new dim (i.e. new cell type in population_b), then existing data has NaNs
    for those.
    """
    if not dataarrays:
        raise ValueError("No DataArrays provided.")

    # 1) collect all dims
    all_dims = set().union(*[da.dims for da in dataarrays])
    unions = {}

    # 2) compute global union of coordinates for each dimension
    for dim in all_dims:
        coords = [
            np.asarray(da[dim].values) for da in dataarrays if dim in da.dims
        ]
        all_coords = np.concatenate(coords)
        if keep_order == "first":
            _, idx = np.unique(all_coords, return_index=True)
            union = all_coords[np.sort(idx)]
        elif keep_order == "sorted":
            union = np.unique(all_coords)
        else:
            raise ValueError("keep_order must be 'first' or 'sorted'")
        unions[dim] = union

    # 3) reindex each DataArray to global union
    reindexed = []
    for da in dataarrays:
        da = da.assign_coords({dim: da[dim].values for dim in da.dims})
        da = da.reindex({dim: unions[dim] for dim in da.dims})
        reindexed.append(da)

    # Use concat along a temporary dummy dimension, then remove it
    combined = xr.concat(reindexed, dim="_tmp_dim").mean(
        dim="_tmp_dim", skipna=True
    )

    return combined

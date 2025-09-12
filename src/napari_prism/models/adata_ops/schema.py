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
import xarray as xr


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
                f"Missing required coorindates: "
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

    This enforces a common metric dimension for all metrics.
    """
    static_dims = ["sample_id", "metric_name", "cell_population_a"]
    # Build coords as (dim, data) pairs or xr.Variable so we can carry attrs
    all_coords = {}

    # sample_id is length-1 along its dim
    all_coords["sample_id"] = ("sample_id", [sample_id])

    # metric_name as a 1-D coordinate with attrs
    all_coords["metric_name"] = xr.Variable(
        ("metric_name",),  # <--- dims tuple
        [metric_name],  # <--- data must be an iterable of length 1
        attrs={"directional": directional, "parameters": parameters},
    )

    # cell_population_a with an attribute stating which column it came fromtype_column": cell_population_a.cell_type_col
    all_coords["cell_population_a"] = cell_population_a.to_xr_variable(
        "cell_population_a"
    )

    if cell_compartment_a is None:
        cell_compartment_a = CompartmentEntity("None", "None")
    all_coords["cell_compartment_a"] = cell_compartment_a.to_xr_variable(
        "cell_compartment_a"
    )
    static_dims += ["cell_compartment_a"]

    if cell_population_b is None:
        cell_population_b = CellEntity("None", "None")
    all_coords["cell_population_b"] = cell_population_b.to_xr_variable(
        "cell_population_b"
    )
    static_dims += ["cell_population_b"]

    if cell_compartment_b is None:
        cell_compartment_b = CompartmentEntity("None", "None")
    all_coords["cell_compartment_b"] = cell_compartment_b.to_xr_variable(
        "cell_compartment_b"
    )
    static_dims += ["cell_compartment_b"]

    all_dims = static_dims + (dims if dims else [])

    # user-supplied extra coords (e.g. {"r": [5, 10]}). For safety wrap them as (dim, data).
    if coords:
        for k, v in coords.items():
            # allow the user to supply either a (dims, data) tuple or raw list
            if (
                isinstance(v, tuple)
                and len(v) == 2
                and isinstance(v[0], list | tuple)
            ):
                # user gave something like (("r",), [5,10]) -> leave as-is
                all_coords[k] = v
            else:
                all_coords[k] = (k, v)

    # Prepare data: left-pad values with singleton dims so it matches len(all_dims)
    arr = np.asarray(values)
    target_ndim = len(all_dims)
    if arr.ndim > target_ndim:
        raise ValueError(
            f"values has too many dimensions: values.ndim={arr.ndim} > target dims={target_ndim}"
        )
    pad = target_ndim - arr.ndim
    arr = arr.reshape((1,) * pad + arr.shape)

    da = xr.DataArray(
        data=arr,
        dims=all_dims,
        coords=all_coords,
    )

    if sample_id_key is not None:
        da["sample_id"].attrs["original_key"] = sample_id_key

    da = da.metric.validate()

    return da

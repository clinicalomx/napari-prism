"""
Dev Notes;
- For xarray integration, should consider composition over inheritance.
- Spatial metrics can be sought to be a multiple inherit. problem;
    - Can be pairwise, self or not
    - Can be compartmental or not
    - Can have xs, ys or not
    etc

"""
from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class CompartmentEntity:
    """Represents a compartment in the tissue."""

    compartment_name: str

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.compartment_name}"

    def __eq__(self, value):
        return self.compartment_name == value.compartment_name


@dataclass(frozen=True)
class CellEntity:
    """Represents a population of a particular cell type"""

    cell_type: str
    compartment: CompartmentEntity | None = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        if self.compartment:
            return f"{self.cell_type}@{self.compartment}"
        return self.cell_type

    @classmethod
    def from_repr(cls, string_repr: str):
        cell_type, compartment = string_repr.split("@")
        compartment = CompartmentEntity(compartment)
        return cls(cell_type, compartment)

# calling dataset.spatial_metric.method
@xr.register_dataarray_accessor("spatial_metric")
class SpatialMetricAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def validate(self):
        """ Validate SpatialMetric schema """
        required_coords = {
            "metric_name",
            "cell_population_a",
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
        """ Nicer representation """
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
        return (
            f"SpatialMetric {metric_name} "
            f"[{pair}]"
        )

def create_spatial_metric(
    values: np.ndarray,
    sample_id: str,
    dims: list[str],
    coords: dict[str, Any],
    cell_population_a: CellEntity,
    cell_population_b: CellEntity | None = None,
    metric_name: str = "generic",
    directional: bool = False,
    parameters: dict[str, Any] | None = None
) -> xr.DataArray:
    """
    Create a SpatialMetric as a DataArray with common metadata.
    """

    # Ensure sample_id is an explicit dimension
    values = np.array([values]) if not dims else np.expand_dims(values, axis=0)
    all_dims = ["sample_id"] + dims
    all_coords = {"sample_id": [sample_id]}
    all_coords.update(coords)

    # Attach population/metric as coords indexed by sample_id
    all_coords["cell_population_a"] = ("sample_id", [str(cell_population_a)])
    all_coords["cell_population_b"] = ("sample_id", [str(cell_population_b)])
    all_coords["metric_name"] = ("sample_id", [metric_name])

    da = xr.DataArray(
        data=values,
        dims=all_dims,
        coords=all_coords,
        attrs={
            "directional": directional,
            "parameters": parameters or {},
        },
    ).spatial_metric.validate()

    return da


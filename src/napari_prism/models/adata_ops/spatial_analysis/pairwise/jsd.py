import itertools as it
from typing import Any

import anndata as ad
import numpy as np
import xarray as xr
from scipy.integrate import simpson
from sklearn.neighbors import KernelDensity

from napari_prism.models.adata_ops.schema import (
    CellEntity,
    CompartmentEntity,
    create_metric,
)


@xr.register_dataarray_accessor("jsd")
class JSDMetricAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def validate(self):
        """
        Validate the JSD metric object.
        """
        self._obj.metric.validate()
        cell_population_a = self._obj.coords.get("cell_population_a", None)
        cell_population_b = self._obj.coords.get("cell_population_b", None)

        if cell_population_b is None:
            raise AttributeError(
                "Need a second cell population for JSD metrics."
            )

        def _retrieve_compartment(label):
            if "@" in label:
                return label.split("@")[-1]
            else:
                return None

        compartment_a = _retrieve_compartment(cell_population_a)
        compartment_b = _retrieve_compartment(cell_population_b)

        if compartment_a != compartment_b:
            raise ValueError(
                "For now, must compare the same compartmental scope."
            )
        return self._obj

    def plot(self):
        self._obj.metric.plot()

    def pretty_print(self):
        self._obj.metric.pretty_print()


def create_jsd_metric(
    values: np.ndarray,
    cell_population_a: CellEntity,
    cell_population_b: CellEntity,
    sample_id: str,
    cell_compartment_a: CompartmentEntity | None = None,
    cell_compartment_b: CompartmentEntity | None = None,
    parameters: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Create a JSD metric as a DataArray.

    Args:
        values: The JSD values
        cell_population_a: Query cell population
        cell_population_b: Target cell population
        sample_id: Sample identifier
        parameters: Additional parameters
    """
    return create_metric(
        values=values,
        sample_id=sample_id,
        dims=[],
        coords={},
        cell_population_a=cell_population_a,
        cell_compartment_a=cell_compartment_a,
        cell_population_b=cell_population_b,
        cell_compartment_b=cell_compartment_b,
        metric_name="jsd",
        directional=False,
        parameters=parameters,
    ).jsd.validate()


def compute_kde(
    spatial_coordinates: np.ndarray, bandwidth=80.0
) -> KernelDensity | None:
    """
    Compute Kernel Density Estimation for given cell coordinates. By default
    with a Gaussian kernel. If given coordinates do not have enough points,
    4 by default, returns None.

    Args:
        spatial_coordinates: N x 2 array of spatial coordinates
        bandwidth: Bandwidth for KDE; equivalent to sigma

    Returns:
        Fitted KernelDensity object or None
    """
    if len(spatial_coordinates) > 4:
        return KernelDensity(
            kernel="gaussian",
            bandwidth=bandwidth,
            algorithm="kd_tree",
            leaf_size=100,
            atol=1e-9,
        ).fit(spatial_coordinates)
    else:
        return None


def _jsd_kdes(
    spatial_coordinates: np.ndarray,
    query_indices: np.ndarray | list,
    target_indices: np.ndarray | list,
    bandwidth=80.0,
) -> float:
    """
    Compute the Jensen-Shannon Divergence between two sets of points modelled
    as Gaussian distributions.

    Returns:
        JSD value between the target and query distributions
    """
    xs = spatial_coordinates[:, 0]
    ys = spatial_coordinates[:, 1]
    xmin = xs.min()
    xmax = xs.max()
    ymin = ys.min()
    ymax = ys.max()

    xseg = np.arange(xmin, xmax, 15)
    yseg = np.arange(ymin, ymax, 15)
    seg_shape = (len(xseg), len(yseg))
    # (self.Xmax - self.Xmin) / 200.0

    x_grid, y_grid = np.meshgrid(xseg, yseg)
    xy_grid = np.vstack([x_grid.flatten(), y_grid.flatten()]).T

    query_kde = compute_kde(
        spatial_coordinates[query_indices], bandwidth=bandwidth
    )

    target_kde = compute_kde(
        spatial_coordinates[target_indices], bandwidth=bandwidth
    )

    query_intensity = np.exp(
        query_kde.score_samples(xy_grid).reshape(seg_shape)
    )

    target_intensity = np.exp(
        target_kde.score_samples(xy_grid).reshape(seg_shape)
    )

    eps = 1e-20
    term1 = 2.0 * query_intensity / (query_intensity + target_intensity + eps)
    term1[np.isnan(term1) | (term1 < eps)] = eps
    term1 = query_intensity * np.log2(term1)
    term1[np.isnan(term1) | (term1 < eps)] = eps

    term2 = 2.0 * target_intensity / (query_intensity + target_intensity + eps)
    term2[np.isnan(term2) | (term2 < eps)] = eps
    term2 = target_intensity * np.log2(term2)
    term2[np.isnan(term2) | (term2 < eps)] = eps

    term1 = simpson(simpson(term1, x=yseg), x=xseg)
    term2 = simpson(simpson(term2, x=yseg), x=xseg)

    # print(f"\tJSD is {np.sqrt(term1 / 2.0 + term2 / 2.0)}")
    return float(np.sqrt(term1 / 2.0 + term2 / 2.0))


def _jsd_single_sample_single_pair(
    adata: ad.AnnData,
    sample_key: str,
    sample_name: str,
    spatial_key: np.ndarray,
    cell_type_key: str,
    query_cell_type: str,
    target_cell_type: str,
    bandwidth: float = 80.0,
    compartment_key: str | None = None,
    cell_compartment: str | None = None,
    minimum_cell_count: int = 10,
):
    adata_subset = adata[adata.obs[sample_key] == sample_name]
    spatial_coordinates = adata_subset.obsm[spatial_key]
    # iloc indexing
    query = np.where(adata_subset.obs[cell_type_key] == query_cell_type)
    query_ix = query[0]
    target = np.where(adata_subset.obs[cell_type_key] == target_cell_type)
    target_ix = target[0]

    if cell_compartment is not None:
        compartment_ix = np.where(
            adata_subset.obs[compartment_key] == cell_compartment
        )[0]
        query_ix = np.intersect1d(query_ix, compartment_ix)
        target_ix = np.intersect1d(target_ix, compartment_ix)

    if (
        len(query_ix) < minimum_cell_count
        or len(target_ix) < minimum_cell_count
    ):
        return np.nan

    return _jsd_kdes(
        spatial_coordinates,
        query_indices=query_ix,
        target_indices=target_ix,
        bandwidth=bandwidth,
    )


def jsd(
    adata: ad.AnnData,
    sample_key: str,
    cell_type_key: str,
    spatial_key: str = "spatial",
    bandwidth: float = 80.0,
    query_cell_types: str | list[str] | None = None,
    target_cell_types: str | list[str] | None = None,
    compartment_key: str | None = None,
    cell_compartment: str | None = None,
    n_processes: int = 1,
    minimum_cell_count: int = 5,  # Minimum cells needed to compute KDE
    as_xarray: bool = True,
):
    """
    Compute Jensen-Shannon Divergence between spatial distributions of cell
    populations.

    Args:
        adata: AnnData object containing spatial data
        sample_key: Column in obs containing sample identifiers
        cell_type_key: Column in obs containing cell type annotations
        spatial_key: Key in obsm containing spatial coordinates
        bandwidth: Bandwidth for KDE estimation
        query_cell_types: Specific cell types to use as query (None for all)
        target_cell_types: Specific cell types to use as target (None for all)
        compartment_key: Column in obs containing compartment information
        cell_compartment: Specific compartment for cells
        n_processes: Number of processes for parallelization
        minimum_cell_count: Minimum number of cells required to compute KDE
        as_xarray: Whether to return results as xarray DataArray

    Returns:
        JSD values for cell type pairs across samples
    """
    # Perform for each sample
    samples = adata.obs[sample_key].unique().tolist()
    cell_types = adata.obs[cell_type_key].unique().tolist()

    # Parse combinations
    if isinstance(query_cell_types, str):
        query_cell_types = [query_cell_types]

    if isinstance(target_cell_types, str):
        target_cell_types = [target_cell_types]

    if query_cell_types is None:
        if target_cell_types is None:
            # All unique unordered combinations, including (A, A)
            pairwise_combinations = it.combinations_with_replacement(
                cell_types, 2
            )
        else:
            pairwise_combinations = {
                tuple(sorted((a, b)))
                for a, b in it.product(cell_types, target_cell_types)
            }
        pairwise_combinations = list(pairwise_combinations)
    else:
        if target_cell_types is None:
            # Query is restricted, target is all cell_types
            pairwise_combinations = {
                tuple(sorted((a, b)))
                for a, b in it.product(query_cell_types, cell_types)
            }
        else:
            # Both query and target are restricted
            pairwise_combinations = {
                tuple(sorted((a, b)))
                for a, b in it.product(query_cell_types, target_cell_types)
            }

    pairwise_combinations = list(pairwise_combinations)

    jsd_results = {}

    if n_processes == 1:
        for sample in samples:
            jsd_results[sample] = {}
            for query_cell_type, target_cell_type in pairwise_combinations:
                val = _jsd_single_sample_single_pair(
                    adata=adata,
                    sample_key=sample_key,
                    sample_name=sample,
                    spatial_key=spatial_key,
                    cell_type_key=cell_type_key,
                    query_cell_type=query_cell_type,
                    target_cell_type=target_cell_type,
                    bandwidth=bandwidth,
                    compartment_key=compartment_key,
                    cell_compartment=cell_compartment,
                    minimum_cell_count=minimum_cell_count,
                )
                jsd_results[sample][(query_cell_type, target_cell_type)] = val
    else:
        from multiprocessing import Pool

        args = [
            (
                adata,  # faster serialisation; subset handled by worker
                sample_key,
                sample_name,  # 2
                spatial_key,
                cell_type_key,
                query_cell_type,  # 5
                target_cell_type,  # 6
                bandwidth,
                compartment_key,
                cell_compartment,
                minimum_cell_count,
            )
            for sample_name in samples
            for query_cell_type, target_cell_type in pairwise_combinations
        ]

        with Pool(processes=n_processes) as pool:
            results = pool.starmap(_jsd_single_sample_single_pair, args)

        jsd_results = {
            (args_tuple[2], args_tuple[5], args_tuple[6]): result
            for args_tuple, result in zip(args, results, strict=False)
        }

    if not as_xarray:
        return jsd_results
    else:
        results = []
        comp = None
        if cell_compartment:
            comp = CompartmentEntity(cell_compartment, compartment_key)

        for patient_id, v in jsd_results.items():
            patient_id = str(patient_id)
            for ct_p, jsd_val in v.items():
                a, b = ct_p
                a_instance = CellEntity(a, cell_type_key)
                b_instance = CellEntity(b, cell_type_key)
                results.append(
                    create_jsd_metric(
                        values=np.array([jsd_val]),
                        cell_population_a=a_instance,
                        cell_population_b=b_instance,
                        cell_compartment_a=comp,
                        cell_compartment_b=comp,
                        sample_id=patient_id,
                        parameters={
                            "bandwidth": bandwidth,
                            "minimum_cell_count": minimum_cell_count,
                        },
                    )
                )

        together = xr.combine_by_coords(results, combine_attrs="no_conflicts")

        return together

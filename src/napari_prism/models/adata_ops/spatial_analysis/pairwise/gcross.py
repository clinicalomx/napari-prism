"""
Dev Notes;
- Gcross -> Can have samples, ncellsa, ncellsb, radii as one single DataArray
- Alternatively can have one cell type pair as one DataArray
- Then all comparisons as one Dataset
"""

import itertools as it
from typing import Any

import anndata as ad
import numpy as np
import xarray as xr
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors

from napari_prism.models.adata_ops.schema import (
    CellEntity,
    CompartmentEntity,
    create_metric,
)
from napari_prism.models.adata_ops.spatial_analysis.graph import (
    get_closest_neighbors,
)


@xr.register_dataarray_accessor("gcross")  # potentially 'distribution'
class GCrossMetricAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def validate(self):
        """
        a) Ensure there's a second population to compare against.
        b) Enforce exclusively global vs global or compartment vs compartment
            comparisons.
        """
        self._obj.metric.validate()
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
        self._obj.metric.plot()

    def pretty_print(self):
        self._obj.metric.pretty_print()

    def auc(self, save_original=False):
        da = self._obj.gcross.validate().copy()

        # Extract radius coordinate for integration
        xs = da.coords["radius"]

        # Apply AUC along radius dimension
        auc_values = xr.apply_ufunc(
            compute_auc_1d,
            da,
            input_core_dims=[["radius"]],
            kwargs={"xs": xs},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        auc_values = auc_values.assign_coords(metric_name=["gcross_auc"])

        auc_values = auc_values.assign_coords(metric_name=["gcross_auc"])

        if "gcross_parameters" in da.coords["metric_name"].attrs:
            auc_values.coords["metric_name"].attrs["gcross_parameters"] = (
                da.coords["metric_name"].attrs["gcross_parameters"]
            )

        return auc_values

    def query(self):
        pass

    def test(self, adata):
        """Test if GCross curve is beyond some homogenous process."""
        indices = self._obj.sample_id  # noqa: F841
        # convex hull;
        # estimate lambda / density
        # theoretical G under given Xs / radii
        # Test observed G vs theoretical G


def homogeneous_gcross_theoretical_values(
    ct_propotions, roi_radius, mean_num_cells, gcross_radii
):
    """
    Calculates theoretical values of gcross for a homogeneous poisson process.

    By C.J.

    Parameters:
    - ct_proportions (list of float): Proportion of cell types. Length gives number of types
    - roi_radius (float): Radius of the circular domain (centered in the middle of the square).
    - mean_num_cells (float): Average number of cells in the ROI.
    - gcross_radii (list of float): Radii the gcross has been calculated at.

    Returns:
    - theoretical_values (np.array): Matrix of theoretical values for given process proportions
        and gcross radii. Shape is (num of gcross radii, num of cell types).
    """

    area_circle = np.pi * (roi_radius**2)
    ct_propotions = np.array(ct_propotions)
    unit_rates = (
        ct_propotions * mean_num_cells / area_circle
    )  # Expected number of cells in circle

    theoretical_value_exponent = (
        -np.expand_dims(unit_rates, axis=0)
        * (np.expand_dims(gcross_radii, axis=1) ** 2)
        * np.pi
    )
    theoretical_values = 1 - np.exp(theoretical_value_exponent)
    return theoretical_values


def create_gcross_metric(
    radii: np.ndarray,
    values: np.ndarray,
    cell_population_a: CellEntity,
    cell_population_b: CellEntity,
    sample_id: str,
    cell_compartment_a: CompartmentEntity | None = None,
    cell_compartment_b: CompartmentEntity | None = None,
    radii_units: str | None = None,
    parameters: dict[str, Any] | None = None,
    sample_id_key: str | None = None,
) -> xr.DataArray:
    """
    Create a GCross metric as a DataArray.

    Args:
        radii: The radius values
        values: The GCross values (same length as radii)
        cell_population_a: Query cell population
        cell_population_b: Target cell population
        parameters: Additional parameters
    """
    assert len(radii) == len(values), "Radii and values must have same length"
    # values = np.array(values).reshape(1, 1, 1, 1, -1)
    return create_metric(
        values=values,
        sample_id=sample_id,
        sample_id_key=sample_id_key,
        dims=["radius"],
        coords={
            "radius": xr.Variable(
                ("radius",), radii, attrs={"units": radii_units}
            )
        },
        cell_population_a=cell_population_a,
        cell_compartment_a=cell_compartment_a,
        cell_population_b=cell_population_b,
        cell_compartment_b=cell_compartment_b,
        metric_name="gcross",
        directional=True,
        parameters=parameters,
    ).gcross.validate()


# single sample func
def _gcross_single_sample_single_pair(
    adata: ad.AnnData,
    sample_key: str,
    sample_name: str,
    spatial_key: np.ndarray,
    cell_type_key: str,
    query_cell_type: str,
    target_cell_type: str,
    compartment_key: str | None = None,
    query_cell_compartment: str | None = None,
    target_cell_compartment: str | None = None,
    max_radius: int | float = 1000,
    num_segments: int = 50,
    correction: str | None = None,
    num_interpoints: int = 100,
    minimum_cell_count: int = 10,  # If either query or target cell type has less than this number of cells, skip
):
    adata_subset = adata[adata.obs[sample_key] == sample_name]
    spatial_coordinates = adata_subset.obsm[spatial_key]

    # iloc indexing
    query_ix = np.where(adata_subset.obs[cell_type_key] == query_cell_type)[0]
    target_ix = np.where(adata_subset.obs[cell_type_key] == target_cell_type)[
        0
    ]

    if (query_cell_compartment is None) != (target_cell_compartment is None):
        raise ValueError(
            "Cannot compare global populations to compartmental" "populations"
        )

    if query_cell_compartment is not None:
        if compartment_key is None:
            raise ValueError(
                "Provide a key or column in .obs for compartment labels."
            )
        else:
            compartment_query_ix = np.where(
                adata_subset.obs[compartment_key] == query_cell_compartment
            )[0]
            compartment_target_ix = np.where(
                adata_subset.obs[compartment_key] == target_cell_compartment
            )[0]
            query_ix = np.intersect1d(query_ix, compartment_query_ix)
            target_ix = np.intersect1d(target_ix, compartment_target_ix)

    # Check if
    if (
        len(query_ix) < minimum_cell_count
        or len(target_ix) < minimum_cell_count
    ):
        # print(f"Skipping {query_cell_type} vs {target_cell_type} in sample {adata_subset.obs[sample_key].unique()[0]} due to insufficient cell count.")
        null_result = np.zeros(num_segments + 1) * np.nan
        return np.linspace(0, max_radius, num_segments + 1), null_result

    if correction is None:
        xs, g_cross = gcross_subset(
            spatial_coordinates=spatial_coordinates,
            query_indices=query_ix,
            target_indices=target_ix,
            max_radius=max_radius,
            num_segments=num_segments,
        )
    elif correction == "border":
        xs, g_cross = gcross_subset_border_correction(
            spatial_coordinates=spatial_coordinates,
            query_indices=query_ix,
            target_indices=target_ix,
            max_radius=max_radius,
            num_segments=num_segments,
            num_interpoints=num_interpoints,
        )
    elif correction == "hanisch_unbiased":
        xs, g_cross = gcross_subset_hanisch_unbiased_correction(
            spatial_coordinates=spatial_coordinates,
            query_indices=query_ix,
            target_indices=target_ix,
            max_radius=max_radius,
            num_segments=num_segments,
            num_interpoints=num_interpoints,
        )
    elif correction == "km":
        xs, g_cross = gcross_subset_kaplan_meier_correction(
            spatial_coordinates=spatial_coordinates,
            query_indices=query_ix,
            target_indices=target_ix,
            max_radius=max_radius,
            num_interpoints=num_interpoints,
        )
    else:
        raise NotImplementedError()

    return xs, g_cross


def gcross(
    adata: ad.AnnData,
    sample_key: str,
    cell_type_key: str,
    spatial_key: str = "spatial",
    spatial_units: str | None = None,
    max_radius: int | float = 100,
    num_segments: int = 50,
    correction: str | None = None,
    num_interpoints: int = 100,
    query_cell_types: str | list[str] | None = None,
    target_cell_types: str | list[str] | None = None,
    compartment_key: str | None = None,
    query_cell_compartment: str | None = None,
    target_cell_compartment: str | None = None,
    n_processes: int = 1,
    minimum_cell_count: int = 10,  # If either query or target cell type has less than this number of cells, skip
    as_xarray: bool = True,  # To return obj or not
):
    # Perform for each sample;
    samples = adata.obs[sample_key].unique().tolist()
    cell_types = adata.obs[cell_type_key].unique().tolist()

    # Parse combinations
    if isinstance(query_cell_types, str):
        query_cell_types = [query_cell_types]

    if isinstance(target_cell_types, str):
        target_cell_types = [target_cell_types]

    if query_cell_types is None:
        if target_cell_types is None:
            # All combinations, directional
            pairwise_combinations = it.product(cell_types, repeat=2)
        else:
            # Combinations with target cell types
            pairwise_combinations = it.product(cell_types, target_cell_types)
    else:
        if target_cell_types is None:
            # Combinations with query cell types
            pairwise_combinations = it.product(query_cell_types, cell_types)
        else:
            # Combinations with specified query and target cell types
            pairwise_combinations = it.product(
                query_cell_types, target_cell_types
            )

    pairwise_combinations = list(pairwise_combinations)

    # Perform for loop or parallelised loop
    g_cross_results = {}

    if n_processes == 1:
        for sample in samples:
            g_cross_results[sample] = {}
            for query_cell_type, target_cell_type in pairwise_combinations:
                xs, g_cross = _gcross_single_sample_single_pair(
                    adata=adata,
                    sample_key=sample_key,
                    sample_name=sample,
                    spatial_key=spatial_key,
                    cell_type_key=cell_type_key,
                    query_cell_type=query_cell_type,
                    target_cell_type=target_cell_type,
                    compartment_key=compartment_key,
                    query_cell_compartment=query_cell_compartment,
                    target_cell_compartment=target_cell_compartment,
                    max_radius=max_radius,
                    num_segments=num_segments,
                    correction=correction,
                    minimum_cell_count=minimum_cell_count,
                    num_interpoints=num_interpoints,
                )
                g_cross_results[sample][
                    (query_cell_type, target_cell_type)
                ] = (xs, g_cross)
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
                compartment_key,
                query_cell_compartment,
                target_cell_compartment,
                max_radius,
                num_segments,
                correction,
                num_interpoints,
                minimum_cell_count,
            )
            for sample_name in samples
            for query_cell_type, target_cell_type in pairwise_combinations
        ]

        with Pool(processes=n_processes) as pool:
            results = pool.starmap(_gcross_single_sample_single_pair, args)

        g_cross_results = {
            (args_tuple[2], args_tuple[5], args_tuple[6]): result
            for args_tuple, result in zip(args, results, strict=False)
        }

    if not as_xarray:
        return g_cross_results
    else:
        results = []
        q_comp = None
        if query_cell_compartment:
            q_comp = CompartmentEntity(query_cell_compartment, compartment_key)
        t_comp = None
        if target_cell_compartment:
            t_comp = CompartmentEntity(
                target_cell_compartment, compartment_key
            )
        for patient_id, v in g_cross_results.items():
            patient_id = str(patient_id)
            for ct_p, res in v.items():
                a, b = ct_p
                a_instance = CellEntity(a, cell_type_key)
                b_instance = CellEntity(b, cell_type_key)
                xs, gc = res
                results.append(
                    create_gcross_metric(
                        radii=xs,
                        values=gc,
                        cell_population_a=a_instance,
                        cell_compartment_a=q_comp,
                        cell_population_b=b_instance,
                        cell_compartment_b=t_comp,
                        sample_id=patient_id,
                        sample_id_key=sample_key,
                        radii_units=spatial_units,
                        parameters={
                            "correction": correction,
                            "minimum_cell_count": minimum_cell_count,
                        },
                    )
                )

        together = xr.combine_by_coords(results, combine_attrs="no_conflicts")

        return together


def gcross_subset(
    spatial_coordinates: np.ndarray,
    query_indices: np.ndarray | list,
    target_indices: np.ndarray | list,
    max_radius: int | float,
    num_segments: int = 50,
):
    """
    Given a M x 1 distance matrix where M represents the query cell type, and
    1 represents its closest neighbour, compute the basic g_cross function.
    """
    distances, indices = get_closest_neighbors(
        spatial_coordinates,
        query_indices=query_indices,
        target_indices=target_indices,
    )

    bins = np.linspace(0, max_radius, num_segments + 1)
    g_cross = np.array([np.mean(distances <= r) for r in bins])
    return bins, g_cross


def gcross_subset_border_correction(
    spatial_coordinates: np.ndarray,
    query_indices: np.ndarray | list,
    target_indices: np.ndarray | list,
    max_radius: int | float,
    num_segments: int = 50,
    num_interpoints: int = 100,
):
    """
    Given a M x 1 distance matrix where M represents the query cell type, and
    1 represents its closest neighbour, compute the basic g_cross function
    with border correction.

    Implementation by C.J
    """
    distances, indices = get_closest_neighbors(
        spatial_coordinates,
        query_indices=query_indices,
        target_indices=target_indices,
    )

    distances_to_edge, _ = get_distances_to_convex_hull_edge(
        spatial_coordinates,
        query_indices=query_indices,
        num_interpoints=num_interpoints,
    )

    bins = np.linspace(0, max_radius, num_segments + 1)
    g_cross = []
    for r in bins:
        neighbor_distances_within_r = distances <= r  # f(w, t)
        r_within_edge_dists = r < distances_to_edge  # f(t, b)

        # f(w, t) x f(t, b)
        numerator = neighbor_distances_within_r & r_within_edge_dists

        # f(w, t) x f(w, b) / sum all f(t, b)
        if r_within_edge_dists.sum() <= 0:
            # If no distances to edge are within the radius, set to 0
            g_cross.append(0.0)
        else:
            g_cross.append(numerator.sum() / r_within_edge_dists.sum())

    return bins, np.array(g_cross)


def gcross_subset_hanisch_unbiased_correction(
    spatial_coordinates: np.ndarray,
    query_indices: np.ndarray | list,
    target_indices: np.ndarray | list,
    max_radius: int | float,
    num_segments: int = 50,
    num_interpoints: int = 100,
):
    """
    Given a M x 1 distance matrix where M represents the query cell type, and
    1 represents its closest neighbour, compute the basic g_cross function
    with hanisch unbiased correction.

    Implementation by C.J
    """
    distances, indices = get_closest_neighbors(
        spatial_coordinates,
        query_indices=query_indices,
        target_indices=target_indices,
    )

    distances_to_edge, _ = get_distances_to_convex_hull_edge(
        spatial_coordinates,
        query_indices=query_indices,
        num_interpoints=num_interpoints,
    )

    areas = get_hanisch_area(
        spatial_coordinates[query_indices, :], distances_to_edge
    )

    # Set any area that is 0 to np inf to account for 0/0
    areas = np.where(areas == 0, np.inf, areas)

    # Numerator
    # f(w, t) -> Distance to nearest neighbour less thna current radius in it
    # f(w, b) > Distance to nearest neighbour less than distance to edge
    # AND GATE for above
    # p(w) -> Area for nn distances

    bins = np.linspace(0, max_radius, num_segments + 1)
    g_cross = []
    for r in bins:
        # f(w, t)
        neighbor_distances_within_r = distances <= r
        # f(w, b)
        neighbor_distances_within_edge_dists = distances < distances_to_edge

        # f(w, t) x f(w, b)
        numerator = (
            neighbor_distances_within_r & neighbor_distances_within_edge_dists
        )

        # Sum [ f(w, t) x f(w, b) / p(w) ]
        numerator = np.sum(numerator / areas)

        # Sum [ f(w, b) / p(w) ]
        if areas.sum() == 0:
            # If no areas are available, set to 0
            denominator = 0.0
        else:
            # f(w, b) -> Distance to nearest neighbour less than distance to edge
            denominator = np.sum(neighbor_distances_within_edge_dists / areas)

        if denominator == 0:
            # If denominator is 0, set to 0
            g_cross.append(0.0)
        else:
            # sum [ f(w, t) x f(w, b) x p(w) ]
            g_cross.append(numerator / denominator)

    return bins, np.array(g_cross)


def gcross_subset_kaplan_meier_correction(
    spatial_coordinates: np.ndarray,
    query_indices: np.ndarray | list,
    target_indices: np.ndarray | list,
    max_radius: int | float,
    num_segments: int = 50,
    num_interpoints: int = 100,
):
    """
    Given a M x 1 distance matrix where M represents the query cell type, and
    1 represents its closest neighbour, compute the basic g_cross function
    with kaplan meier correction.

    Implementation by C.J
    """
    distances, indices = get_closest_neighbors(
        spatial_coordinates,
        query_indices=query_indices,
        target_indices=target_indices,
    )

    distances_to_edge, _ = get_distances_to_convex_hull_edge(
        spatial_coordinates,
        query_indices=query_indices,
        num_interpoints=num_interpoints,
    )

    bins = np.linspace(0, max_radius, num_segments + 1)

    g_cross = _compute_kaplan_meier_estimate(
        distances, distances_to_edge, bins
    )
    return bins, np.append([0], g_cross)


def _compute_kaplan_meier_estimate(
    distances: np.ndarray | list,
    distances_to_edge: np.ndarray | list,
    bins: np.ndarray | list,
):
    """
    Compute binned kaplan meier estimate given censored data, censoring time
    and bins.

    Implementation by C.J
    """
    # Get minimum of distance
    min_edge_neighbour_distance = np.minimum(distances, distances_to_edge)
    is_uncensored = distances < distances_to_edge

    # Bin both all distances and uncensored distances
    binned_distances, _ = np.histogram(min_edge_neighbour_distance, bins=bins)
    binned_uncensored_distances, _ = np.histogram(
        min_edge_neighbour_distance[is_uncensored], bins=bins
    )

    # Get number of distances above maximum radius. Assumes last bin is
    # the highest radius
    number_distances_above_max = (min_edge_neighbour_distance > bins[-1]).sum()

    # Count of greater uncensored distances
    number_greater_uncensored = (
        binned_distances[::-1].cumsum()[::-1] + number_distances_above_max
    )

    # Zero divide by zero case handled by where so ignoring warning
    # Divide by zero case shouldn't occur so that warning is kept
    with np.errstate(invalid="ignore"):
        km_factors = np.where(
            number_greater_uncensored > 0,
            1 - binned_uncensored_distances / number_greater_uncensored,
            1,
        )

    km_estimate = 1 - np.cumprod(km_factors)

    return km_estimate


def compute_auc_1d(
    values: np.ndarray | list,
    xs: np.ndarray | list | None = None,
    axis: int = -1,
):
    """
    Compute the area under the curve for a 1D distribution.
    """
    if xs is not None:
        return np.trapezoid(y=values, x=xs, axis=axis)
    return np.trapezoid(y=values, axis=axis)


# possible utils
def get_distances_to_convex_hull_edge(
    spatial_coordinates: np.ndarray,
    query_indices: np.ndarray | list | None = None,
    num_interpoints: int = 100,
):
    """
    Get the distances to the convex hull edge for each point in the full
    dataset.

    Args:
        spatial_coordinates (np.ndarray): N x 2 array of spatial coordinates.
        num_interpoints (int): Number of points to interpolate along
            each simplex. Defaults to 50. Note each simplex will have
            2 additional points respresenting the end points of the
            simplex.

    From C.J
    """
    if query_indices is not None:
        spatial_coordinates = spatial_coordinates[query_indices, :]
    hull = ConvexHull(spatial_coordinates)
    x1_ind = hull.simplices[:, 0]
    x2_ind = hull.simplices[:, 1]
    diff = spatial_coordinates[x2_ind] - spatial_coordinates[x1_ind]

    # Add points spaced along simplicies; *
    approxed_outside = np.empty((0, 2))
    for impute_lambda in np.linspace(0, 1, num=2 + num_interpoints):
        approxed_outside = np.append(
            approxed_outside,
            diff * impute_lambda + spatial_coordinates[x1_ind],
            axis=0,
        )

    # no self distances since apply on diff dataset
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(approxed_outside)
    distances, indices = nn.kneighbors(spatial_coordinates, n_neighbors=1)
    return distances[:, 0], indices[:, 0]


def get_hanisch_area(
    spatial_coordinates: np.ndarray, distances_to_edge: np.ndarray
):
    """
    Approximates area of the ROI using a convex hull when cells further
    from the edge are progressively removed. Areas are only required for
    computation of the hanisch unbiased estimator of g-cross.

    From C.J
    """
    areas = []
    for d in distances_to_edge:
        # use the edge distances more than the current radius to compute area;
        uncensored = distances_to_edge >= d

        if uncensored.sum() > 3:
            area = ConvexHull(spatial_coordinates[uncensored, :]).volume
        else:
            area = 0.0
        areas.append(area)

    return np.array(areas)

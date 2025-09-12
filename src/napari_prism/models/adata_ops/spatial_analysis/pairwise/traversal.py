import itertools as it
from typing import Any, Literal

import anndata as ad
import numpy as np
import xarray as xr
from scipy.sparse import issparse

from napari_prism.models.adata_ops.spatial_analysis.schema import (
    CellEntity,
    CompartmentEntity,
    create_spatial_metric,
)


@xr.register_dataarray_accessor("barrier_score")
class BarrierScoreMetricAccessor:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def validate(self):
        self._obj.spatial_metric.validate()
        return self._obj

    def plot(self):
        self._obj.spatial_metric.plot()

    def pretty_print(self):
        self._obj.spatial_metric.pretty_print()


def create_barrier_score_metric(
    values: np.ndarray,
    cell_population_a: CellEntity,
    cell_population_b: CellEntity,
    cell_population_barrier: CellEntity,
    sample_id: str,
    parameters: dict[str, Any] | None = None,
) -> xr.DataArray:
    """
    Create the barrier_score metric as a DataArray.

    Args:
        values: The barrier_score value
        cell_population_a: Query cell population
        cell_population_b: Target cell population
        cell_population_barrier: Barrier cell population
        parameters: Additional parameters
    """
    if parameters is None:
        parameters = {}

    parameters["barrier_cell"] = str(cell_population_barrier)

    return create_spatial_metric(
        values=values,
        sample_id=sample_id,
        dims=[],
        coords={},
        cell_population_a=cell_population_a,
        cell_population_b=cell_population_b,
        metric_name="barrier_score",
        directional=True,
        parameters=parameters,
    ).barrier_score.validate()


def _barrier_score_single_sample(
    adata: ad.AnnData,
    sample_key: str,
    sample_name: str,
    connectivity_key: str,
    cell_type_key: str,
    query_cell_type: str,
    target_cell_type: str,
    barrier_cell_type: str,
    enforce_barrier_next_to_compartment: bool = False,
    compartment_key: str | None = None,
    barrier_cell_compartment: str | None = None,
    query_cell_compartment: str | None = None,
    target_cell_compartment: str | None = None,
    backend: Literal["numpy", "networkx", "cugraph"] = "numpy",
) -> float:
    """
    Compute a barrier score for a single sample and cell type pair.

    Barrier score is the average number of barrier cells along all shortest
    paths from each query cell to the nearest target cell(s).

    Args:
        adata: AnnData
            Single-cell dataset.
        sample_key: str
            Column in .obs indicating sample assignment.
        sample_name: str
            Name of the sample to subset.
        spatial_key: str
            Key in .obsm for spatial coordinates.
        connectivity_key: str
            Key in .obsp for the precomputed neighbor graph (symmetric adjacency).
        cell_type_key: str
            Column in .obs for cell type labels.
        query_cell_type: str
            Cell type for origin cells (e.g., CD8 T cells).
        target_cell_type: str
            Cell type for target cells (e.g., tumor).
        barrier_cell_type: str
            Cell type forming the barrier (e.g., fibroblasts).
        enforce_barrier_next_to_compartment : bool
            If True, only count barrier cells that are adjacent to a target compartment.
        compartment_key: str | None
            Column in .obs for compartment labels.
        query_cell_compartment: str | None
            Compartment label for query cells.
        target_cell_compartment: str | None
            Compartment label for target cells.

    Returns:
        scores_per_query: np.ndarray
            Barrier score per query cell.
        mean_score: float
            Mean barrier score across all query cells.
    """

    adata_subset = adata[adata.obs[sample_key] == sample_name]

    # select cell types
    query_ix = np.where(adata_subset.obs[cell_type_key] == query_cell_type)[0]
    target_ix = np.where(adata_subset.obs[cell_type_key] == target_cell_type)[
        0
    ]
    barrier_ix = np.where(
        adata_subset.obs[cell_type_key] == barrier_cell_type
    )[0]

    # filter by compartments
    if (
        query_cell_compartment is not None
        or target_cell_compartment is not None
        or barrier_cell_compartment is not None
    ):
        if compartment_key is None:
            raise ValueError(
                "Provide a compartment_key for compartment filtering"
            )
        if query_cell_compartment:
            query_comp_ix = np.where(
                adata_subset.obs[compartment_key] == query_cell_compartment
            )[0]
            query_ix = np.intersect1d(query_ix, query_comp_ix)
        if target_cell_compartment:
            target_comp_ix = np.where(
                adata_subset.obs[compartment_key] == target_cell_compartment
            )[0]
            target_ix = np.intersect1d(target_ix, target_comp_ix)
        if barrier_cell_compartment:
            barrier_comp_ix = np.where(
                adata_subset.obs[compartment_key] == barrier_cell_compartment
            )
            barrier_ix = np.intersect1d(barrier_ix, barrier_comp_ix)

    if len(query_ix) == 0 or len(target_ix) == 0:
        return np.nan  # nothing to compute

    adjacency = adata_subset.obsp[connectivity_key]

    if backend == "numpy":
        if issparse(adjacency):
            adjacency = adjacency.tolil()  # easier row-based iteration

        # optionally restrict barrier cells to those next to target compartment
        if enforce_barrier_next_to_compartment and compartment_key is not None:
            neighbor_matrix = adjacency.copy()
            if issparse(neighbor_matrix):
                neighbor_matrix = neighbor_matrix.tolil()
            valid_barrier_ix = set()
            for b in barrier_ix:
                neighbors = (
                    neighbor_matrix.rows[b]
                    if issparse(neighbor_matrix)
                    else np.where(neighbor_matrix[b])[0]
                )
                if any(n in target_ix for n in neighbors):
                    valid_barrier_ix.add(b)
            barrier_ix = np.array(list(valid_barrier_ix))

        # BFS traversal for each query cell
        scores_per_query = []
        for q in query_ix:
            scores = []
            visited = {q}
            frontier = [(q, 0)]  # (node, barrier_count)

            while frontier:
                node, bcount = frontier.pop(0)
                if node in target_ix:
                    scores.append(bcount)
                    continue
                neighbors = (
                    adjacency.rows[node]
                    if issparse(adjacency)
                    else np.where(adjacency[node])[0]
                )
                for neighbor in neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        frontier.append(
                            (
                                neighbor,
                                bcount + (1 if neighbor in barrier_ix else 0),
                            )
                        )

            scores_per_query.append(np.mean(scores) if scores else 0.0)

        scores_per_query = np.array(scores_per_query)
        return np.mean(scores_per_query)

    else:
        raise NotImplementedError("Other backends not implemented yet.")


def barrier_score(
    adata: ad.AnnData,
    sample_key: str,
    cell_type_key: str,
    connectivity_key: str,
    barrier_cell_type: str,  # usually fibroblasts / maybe tumors with some expression
    query_cell_types: str | list[str] | None = None,
    target_cell_types: str | list[str] | None = None,
    compartment_key: str | None = None,
    barrier_cell_compartment: str | None = None,
    query_cell_compartment: str | None = None,
    target_cell_compartment: str | None = None,
    enforce_barrier_next_to_compartment: bool = False,
    n_processes: int = 1,
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
    barrier_score_results = {}
    if n_processes == 1:
        for sample in samples:
            barrier_score_results[sample] = {}
            for query_cell_type, target_cell_type in pairwise_combinations:
                barrier_score = _barrier_score_single_sample(
                    adata=adata,
                    sample_key=sample_key,
                    sample_name=sample,
                    connectivity_key=connectivity_key,
                    cell_type_key=cell_type_key,
                    query_cell_type=query_cell_type,
                    target_cell_type=target_cell_type,
                    barrier_cell_type=barrier_cell_type,
                    compartment_key=compartment_key,
                    barrier_cell_compartment=barrier_cell_compartment,
                    query_cell_compartment=query_cell_compartment,
                    target_cell_compartment=target_cell_compartment,
                    enforce_barrier_next_to_compartment=enforce_barrier_next_to_compartment,
                )
                barrier_score_results[sample][
                    (query_cell_type, target_cell_type)
                ] = barrier_score

    if not as_xarray:
        return barrier_score_results
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
        b_comp = None
        if barrier_cell_compartment:
            b_comp = CompartmentEntity(
                barrier_cell_compartment, compartment_key
            )

        for patient_id, v in barrier_score_results.items():
            patient_id = str(patient_id)
            for ct_p, res in v.items():
                a, b = ct_p
                a_instance = CellEntity(a, cell_type_key, q_comp)
                b_instance = CellEntity(b, cell_type_key, t_comp)
                barrier_instance = CellEntity(
                    barrier_cell_type, cell_type_key, b_comp
                )
                results.append(
                    create_barrier_score_metric(
                        values=res,
                        cell_population_a=a_instance,
                        cell_population_b=b_instance,
                        cell_population_barrier=barrier_instance,
                        sample_id=patient_id,
                        parameters={
                            "enforce_barrier_next_to_component": enforce_barrier_next_to_compartment,
                        },
                    )
                )

        together = xr.combine_by_coords(results, combine_attrs="no_conflicts")
        return together


def probabilistic_traversal():
    pass

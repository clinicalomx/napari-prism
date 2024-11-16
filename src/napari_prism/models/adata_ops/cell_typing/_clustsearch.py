"""Originally from phenotyping.py"""

import gc
import time
from datetime import datetime
from types import MappingProxyType
from typing import Literal, Union

import igraph as ig
import leidenalg as lg
import loguru
import numpy as np
import pandas as pd
import sklearn.neighbors as skn
from anndata import AnnData

# TODO: make this cpu only
from phenograph.core import parallel_jaccard_kernel
from scipy import sparse as sp

from napari_prism.models._utils import overrides


def gpu_import_error_message(package_name: str) -> str:
    """Standard error message when importing a GPU package fails.

    Args:
        package_name: Name of the package that failed to import.

    Returns:
        The formatted error message directing user to install the GPU version of
            the package.
    """
    return (
        f"{package_name} not installed. Need to install gpu extras with: "
        "`pip install 'napari-prism[gpu]'`"
    )


class KNN:
    """Base class for computing a K-Nearest Neighbor search."""

    def compute_neighbors():  # data, K, output_type
        raise NotImplementedError("Abstract method.")

    def indices_to_edgelist():
        """Convert indices output from sklearn-like neighbours.kneighbors outputs."""
        raise NotImplementedError("Abstract method.")

    def edgelist_to_graph():
        """Convert edgelist to graph."""
        raise NotImplementedError("Abstract method.")

    def _remove_self_distances(self, d, idx):
        raise NotImplementedError("Abstract method.")

    def _log_equivalent_metrics(self, metric, p) -> tuple[str, str]:
        """Logs the equivalent metric for a given Minowski distance and `p`
        value.

        Args:
            metric: Distance metric.
            p: Power parameter for Minkowski metric.

        Returns:
            Tuple of the metric and the equivalent metric for logging purposes.
        """
        if metric == "minkowski" and p == 2:
            log_metric = "euclidean"
        elif metric == "minkowski" and p == 1:
            log_metric = "manhattan"
        else:
            log_metric = None
        return metric, log_metric


class KNNCPU(KNN):
    """CPU (sklearn) implementation and backend for performing a KNN search."""

    @overrides(KNN)
    def _remove_self_distances(self, d, idx) -> tuple[np.ndarray, np.ndarray]:
        if idx[0, 0] == 0:
            idx = np.delete(idx, 0, axis=1)
            d = np.delete(d, 0, axis=1)
        else:  # Otherwise delete the _last_ column of d and idx
            idx = np.delete(idx, -1, axis=1)
            d = np.delete(d, -1, axis=1)
        return d, idx

    @overrides(KNN)
    def compute_neighbors(
        self,
        data: np.ndarray,
        n_neighbors: int,
        algorithm: str = "auto",  # KNN search method
        metric: str = "minkowski",
        p: int = 2,  # Power parameter for Minkowski metric
        n_jobs: int = -1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the K-Nearest Neighbors search on the CPU.

        Wraps sklearn.neighbors.NearestNeighbors with logging and automatic
        selection of appropriate parameters. Enforces brute force search for
        cosine and correlation metrics. Removes self distances from the output.

        Args:
            data: Input data matrix (N samples, M features).
            n_neighbors: Number of nearest neighbors to compute.
            algorithm: Algorithm to use for nearest neighbors search.
            metric: Distance metric to use for the nearest neighbors search.
            p: Power parameter for Minkowski metric.
            n_jobs: Number of parallel jobs to run.

        Returns:
            Tuple of the distances and indices of the K nearest neighbors.
        """
        # time operation
        subtic = time.time()

        # Log equivalent distance metric for clarity
        metric, log_metric = self._log_equivalent_metrics(metric, p)
        k_string = f"\t K = {n_neighbors}\n"
        distance_string = (
            f"\t Distance Metric = {metric}\n"
            if log_metric is None
            else f"\t Distance Metric = {log_metric} ({metric} backend)\n"
        )
        algorithm_string = f"\t Search Algorithm = {algorithm}\n"

        loguru.logger.info(
            "Performing KNN search on CPU: \n"
            + k_string
            + distance_string
            + algorithm_string,
        )

        # Enforce brute force if metric is cosine or correlation
        if metric in ["cosine", "correlation"]:
            algorithm = "brute"
            loguru.logger.info(f"Enforcing brute search for {metric} metric")

        knn = skn.NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # Due to results including self
            algorithm=algorithm,
            metric=metric,
            p=p,
            n_jobs=n_jobs,
        )

        knn.fit(data)
        d, idx = knn.kneighbors(data)
        d, idx = self._remove_self_distances(d, idx)
        loguru.logger.info(
            f"KNN CPU computed in {time.time() - subtic} seconds \n",
        )
        return d, idx


class KNNGPU(KNN):
    """GPU (cuml) implementation and backend for performing a KNN search."""

    AVAIL_DIST_METRICS = [
        "euclidean",
        "l2",
        "sqeuclidean",
        "cityblock",
        "l1",
        "manhattan",
        "taxicab",
        "braycurtis",
        "canberra",
        "minkowski",
        "lp",
        "chebyshev",
        "linf",
        "jensenshannon",
        "cosine",
        "correlation",
        "inner_product",
        "jaccard",
        "hellinger",
        "haversine",
    ]

    def __init__(self):
        self.check_if_GPU_version_installed()

    def check_if_GPU_version_installed(self):
        """Check if the necessary RAPIDS (GPU) packages are installed.

        Raises:
            ImportError: If the necessary RAPIDS packages are not installed.
                Provides a message directing the user on how to install the
                missing packages.
        """
        try:
            import cuml

            self.cuml = cuml
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cuml")) from e
        try:
            import cupy

            self.cupy = cupy
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cupy")) from e

    @overrides(KNN)
    def _remove_self_distances(self, d, idx) -> tuple[np.ndarray, np.ndarray]:
        if idx[0, 0] == 0:
            idx = self.cupy.delete(idx, 0, axis=1)
            d = self.cupy.delete(d, 0, axis=1)
        else:  # Otherwise delete the _last_ column of d and idx
            idx = self.cupy.delete(idx, -1, axis=1)
            d = self.cupy.delete(d, -1, axis=1)
        return d, idx

    @overrides(KNN)
    def compute_neighbors(
        self,
        data: Union[np.ndarray, "cupy.ndarray"],  # type: ignore # noqa: F821
        n_neighbors: int,
        algorithm: str = "auto",
        metric: str = "euclidean",
        p: int = 2,  # Power parameter for Minkowski metric
        output_type: str = "cupy",  # {‘input’, ‘array’, ‘dataframe’, ‘series’, ‘df_obj’, ‘numba’, ‘cupy’, ‘numpy’, ‘cudf’, ‘pandas’}
        two_pass_precision: bool = False,
    ) -> "tuple[cupy.ndarray, cupy.ndarray]":  # type: ignore # noqa: F821
        """Compute the K-Nearest Neighbors search on the GPU.

        Wraps cuml.neighbors.NearestNeighbors with logging and automatic
        selection of appropriate parameters. Enforces brute force search for
        cosine and correlation metrics. Removes self distances from the output.

        Args:
            data: Input data matrix (N samples, M features).
            n_neighbors: Number of nearest neighbors to compute.
            algorithm: Algorithm to use for nearest neighbors search.
            metric: Distance metric to use for the nearest neighbors search.
            p: Power parameter for Minkowski metric.
            n_jobs: Number of parallel jobs to run.
            output_type: Output type for the distances and indices. Passed to
                cuml.neighbors.NearestNeighbors.
            two_pass_precision: Performs two-pass precision for the KNN search.
                Defaults to False due to a bug in
                https://github.com/rapidsai/cuml/issues/5788

        Returns:
            Tuple of the distances and indices of the K nearest neighbors.
        """
        # time operation
        subtic = time.time()

        # Log equivalent distance metric for clarity
        metric, log_metric = self._log_equivalent_metrics(metric, p)
        k_string = f"\t K = {n_neighbors}\n"
        distance_string = (
            f"\t Distance Metric = {metric}\n"
            if log_metric is None
            else f"\t Distance Metric = {log_metric} ({metric} backend; SLOWER)\n"
        )
        algorithm_string = f"\t Search Algorithm = {algorithm}\n"

        loguru.logger.info(
            "Performing KNN search on GPU: \n"
            + k_string
            + distance_string
            + algorithm_string,
        )

        # Enforce brute force if metric is cosine or correlation
        if metric in ["cosine", "correlation"]:
            algorithm = "brute"
            loguru.logger.info(
                f"Enforcing brute search for {metric} metric", flush=True
            )

        knn = self.cuml.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1,  # Due to results including self
            algorithm=algorithm,
            metric=metric,
            p=2,
            output_type=output_type,
        )
        # Can technically do rsc.pp.neighbors;
        # rsc.pp.neighbors; if brute -> cupy, returns idx, d
        # idx, d --> distances as csr_matrix
        # Cast data to cupy array
        X_cupy = self.cupy.asarray(data)
        knn.fit(X_cupy)
        d, idx = knn.kneighbors(X_cupy, two_pass_precision=two_pass_precision)
        d, idx = self._remove_self_distances(d, idx)
        loguru.logger.info(
            f"KNN GPU computed in {time.time() - subtic} seconds \n"
        )
        return d, idx


class KNNRSC(KNNGPU):
    """GPU (rapids_singlecell) implementation and backend for performing a KNN
    search."""

    def __init__(self):
        self.check_if_GPU_version_installed()

    @overrides(KNNGPU)
    def check_if_GPU_version_installed(self):
        """Check if rapids_singlecell is installed.

        Raises:
            ImportError: If rapids_singlecell is not installed.
                Provides a message directing the user on how to install the
                missing packages.
        """
        try:
            import rapids_singlecell

            self.rsc = rapids_singlecell
        except ImportError as e:
            raise ImportError(
                gpu_import_error_message("rapids-singlecell")
            ) from e

    def _reverse_csr_matrix(self, csr_matrix):
        """Undo the CSR operation of a cupyx sparse csr matrix.
        rsc.pp.neighbors merges d, idx -> weighted adjacency csr
        This converts weighted adjacency csr -> d, idx"""
        rowptr = csr_matrix.indptr
        n_neighbors = (rowptr[1] - rowptr[0]).item()
        flattened_data = csr_matrix.data
        flattened_indices = csr_matrix.indices
        d = flattened_data.reshape((len(rowptr) - 1, n_neighbors))
        idx = flattened_indices.reshape((len(rowptr) - 1, n_neighbors))
        return d, idx

    @overrides(KNNGPU)
    def compute_neighbors(
        self,
        adata: AnnData,
        n_neighbors: int,
        n_pcs: int,
        use_rep: str,
        random_state: int = 0,
        algorithm: str = "auto",
        metric: str = "euclidean",
        metric_kwds: MappingProxyType = MappingProxyType({}),
        key_added: str = "rsc_neighbors",
        output_type: str = "csr",
    ) -> Union[
        "tuple[cupy.ndarray, cupy.ndarray]", "cupyx.scipy.sparse.csr_matrix"  # type: ignore # noqa: F821
    ]:
        """Compute the K-Nearest Neighbors search on the GPU, compatiable
        directly with AnnData objects.

        Uses rapids_singlecell.pp.neighbors to compute the KNN graph. Formats
        the output to the distances and indices of the K nearest neighbors
        similar to the sklearn and cuml implementations.

        Args:
            adata: AnnData object.
            n_neighbors: Number of nearest neighbors to compute.
            n_pcs: Number of principal components to use.
            use_rep: Representation to use.
            random_state: Random seed.
            algorithm: Algorithm to use for nearest neighbors search.
            metric: Distance metric to use for the nearest neighbors search.
            metric_kwds: Additional keyword arguments for the distance metric.
            key_added: Key to add to the AnnData object.
            output_type: Output type for the distances and indices.


        Returns:
            If `output_type` is 'csr', returns the weighted adjcency matrix as a
            cupyx.scipy.sparse.csr_matrix. Otherwise, returns a tuple of the
            distances and indices of the K nearest neighbors.
        """

        # Inplace operations to save connectivities manifold --> UMAP
        # But technically should be iterated over to go over umap over various K's
        # So key added can probably iterate K?
        self.rsc.pp.neighbors(
            adata,
            n_neighbors=n_neighbors,
            n_pcs=n_pcs,
            use_rep=use_rep,
            random_state=random_state,
            algorithm=algorithm,
            metric=metric,
            metric_kwds=metric_kwds,
            key_added=key_added,
            copy=False,
        )

        weighted_adjacency_csr = adata.obsp[
            key_added + "_distances"
        ]  # cupyx.scipy.sparse.csr_matrix

        if output_type == "csr":
            return weighted_adjacency_csr
        else:
            # Return to d, idx format with no self distances in the KNN graph;
            d, idx = self._reverse_csr_matrix(weighted_adjacency_csr)
            d, idx = self._remove_self_distances(d, idx)
            return d, idx


# TODO: Maybe extend link prediction from CuGraph to swap out coefficients;

# Jaccard vs Overlap:
# - Jaccard: O(|A|+|B|), but since K = |A| = |B|, then O(N), across all pairs
# Same for Overlap

# Issue with GPU Jaccard is the large memory demand.
# https://synergy.cs.vt.edu/pubs/papers/sathre-fpga-jaccard-hpec2022.pdf
# https://www.researchgate.net/publication/285835027_GPUSCAN_GPU-based_Parallel_Structural_Clustering_Algorithm_for_Networks
# https://www.nature.com/articles/s43588-023-00465-8

# Memory CuGraph resolve: https://medium.com/rapids-ai/tackling-large-graphs-with-rapids-cugraph-and-unified-virtual-memory-b5b69a065d4
# try with JaccardGPU;
# rmm.reinitialize(managed_memory=True)
# assert(rmm.is_initialized())


class GraphRefiner:
    """Base class for refining a KNN graph by defining edge weights or
    distances with the Jaccard index."""

    def compute_jaccard():
        raise NotImplementedError("Abstract method.")

    def idx_partner_to_edgelist(self, idx):
        """Converts the KNN idx outputs to an edgelist format:
        [[1301, 1500, 5030, 5030], # index 0
         [1500, 1133, 1301, 5030], # index 1
         ...
         [5030, 1313, 1500, 1133]] # index N-1

        [[0, 1301],
         [0, 1500],
         [0, 5030],
         [1, 1500],
         [1, 1133],
         ...
         [N-1, 1133],
         [N-1, 1500],
         [N-1, 1313],
         [N-1, 5030]]
        """
        raise NotImplementedError("Abstract method.")

    def idx_partner_to_coo(self, idx):
        """Converts the KNN idx outputs to an coo format:
            [[1301, 1500, 5030, 5030], # index 0
             [1500, 1133, 1301, 5030], # index 1
             ...
             [5030, 1313, 1500, 1133]] # index N-1

             to

            [[0, 1301],
             [0, 1500],
             [0, 5030],
             [1, 1500],
             [1, 1133],
             ...
             [N-1, 1133],
             [N-1, 1500],
             [N-1, 1313],
             [N-1, 5030]]
              i, j, s = kernel(**kernelargs)
        n, k = kernelargs["idx"].shape
        graph = sp.coo_matrix((s, (i, j)), shape=(n, n))

        """
        raise NotImplementedError("Abstract method.")


class JaccardRefinerCPU(GraphRefiner):
    """CPU (phenograph) implementation and backend for refining a KNN graph"""

    def coo_symmatrix_to_edgelist(self, coo):
        edgelist = np.vstack(coo.nonzero()).T.tolist()
        # For now pd dataframe struct
        edgelist = pd.DataFrame(edgelist)
        edgelist.columns = [
            "first",
            "second",
        ]  # Follow convention of jaccard gpu
        edgelist["jaccard_coeff"] = coo.data
        return edgelist

    @overrides(GraphRefiner)
    def idx_partner_to_edgelist(self, idx):
        return np.column_stack(
            (np.repeat(np.arange(idx.shape[0]), idx.shape[1]), idx.ravel())
        )

    @overrides(GraphRefiner)
    def compute_jaccard(self, idx) -> pd.DataFrame:
        """Compute the Jaccard index on the CPU.

        Args:
            idx: KNN graph indices in the form of a (N samples, K neighbors)
                matrix.

        Returns:
            Edgelist of the Jaccard index.
        """
        subtic = time.time()
        loguru.logger.info(
            f"Performing Jaccard on CPU:\n"
            f"\t KNN graph nodes = {idx.shape[0]}\n"
            f"\t KNN graph K-neighbors = {idx.shape[1]}\n"
        )
        # NetworkX-like format
        # NOTE: Below is a direct-neighbor comparison -> one-hop neighbors
        i, j, s = parallel_jaccard_kernel(idx)
        jaccard_graph = sp.coo_matrix(
            (s, (i, j)), shape=(idx.shape[0], idx.shape[0])
        )
        # Graph is un-directed, so we need to symmetrize;
        jaccard_graph = (jaccard_graph + jaccard_graph.transpose()).multiply(
            0.5
        )
        # retain lower triangle (for efficiency) ---> Assumes symmetry for KNN graph ..?
        jaccard_coo_symmatrix = sp.tril(jaccard_graph, -1)
        jaccard_edgelist = self.coo_symmatrix_to_edgelist(
            jaccard_coo_symmatrix
        )

        jaccard_edgelist = self.add_isolated_nodes(
            idx.shape[0], jaccard_edgelist
        )

        loguru.logger.info(
            f"Jaccard CPU edgelist constructed in {time.time() - subtic}"
            f"seconds \n"
        )
        return jaccard_edgelist

    def add_isolated_nodes(
        self, n_cells: int, edgelist: pd.DataFrame
    ) -> pd.DataFrame:
        """Add isolated nodes to the edgelist to ensure all nodes are included.

        Args:
            n_cells: Number of cells in the dataset.
            edgelist: Edgelist of the KNN graph.
        """
        # Will just return edgelist if none
        g_list = list(set(edgelist["first"]).union(set(edgelist["second"])))
        indices = np.arange(0, n_cells)
        li1 = np.array(g_list)
        li2 = np.array(indices)
        dif1 = np.setdiff1d(li1, li2)
        dif2 = np.setdiff1d(li2, li1)
        isolated_nodes = np.concatenate((dif1, dif2))
        # Add as nodes with self loops
        isolated_nodes_df = pd.DataFrame()
        isolated_nodes_df["first"] = isolated_nodes
        isolated_nodes_df["second"] = isolated_nodes
        isolated_nodes_df["jaccard_coeff"] = np.ones(isolated_nodes.shape)
        # Added nodes
        result = pd.concat(
            [edgelist, isolated_nodes_df], axis=0, ignore_index=True
        )
        return result


class JaccardRefinerGPU(GraphRefiner):
    """GPU (cugraph) implementation and backend for refining a KNN graph."""

    def __init__(self):
        self.check_if_GPU_version_installed()

    @overrides(GraphRefiner)
    def idx_partner_to_edgelist(self, idx, output_type="cupy"):
        if isinstance(idx, self.cupy.ndarray):
            edgelist = self.cupy.column_stack(
                (
                    self.cupy.repeat(
                        self.cupy.arange(idx.shape[0]), idx.shape[1]
                    ),
                    idx.ravel(),
                )
            )
        else:
            raise ValueError("Unsupported idx type. Takes in cupy.ndarray")

        if output_type == "cupy":
            return edgelist

        elif output_type == "cudf":
            cdf = self.cudf.DataFrame(edgelist).rename(
                columns={0: "source", 1: "destination"}
            )
            return cdf
        else:
            raise ValueError("Unsupported type")

    def check_if_GPU_version_installed(self):
        """Check if the necessary RAPIDS (GPU) packages are installed.

        Raises:
            ImportError: If the necessary RAPIDS packages are not installed.
                Provides a message directing the user on how to install the
                missing packages.
        """
        try:
            import cugraph

            self.cugraph = cugraph
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cugraph")) from e

        try:
            import cudf

            self.cudf = cudf
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cudf")) from e

        try:
            import rmm

            self.rmm = rmm
        except ImportError as e:
            raise ImportError(gpu_import_error_message("rmm")) from e

        try:
            import cupy

            self.cupy = cupy
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cupy")) from e

    def initialise_uvm(self):
        """Set RMM to allocate all memory as managed memory; unified virtual memory
        aka GPU + CPU"""
        loguru.logger.info(
            "Initialising rapids memory manager to enable memory"
            "oversubscription with unified virtual memory..."
        )
        self.rmm.mr.set_current_device_resource(
            self.rmm.mr.ManagedMemoryResource()
        )
        assert self.rmm.is_initialized()

    def compute_jaccard(self, idx, two_hop=False):
        self.initialise_uvm()

        subtic = time.time()
        loguru.logger.info(
            f"Performing Jaccard on GPU:\n"
            f"\t KNN graph nodes = {idx.shape[0]}\n"
            f"\t KNN graph K-neighbors = {idx.shape[1]}\n"
        )

        loguru.logger.info(
            (
                "NOTE: This performs undirected KNN Jaccard refinement. "
                "Different to the original CPU implementation, which is a"
                " directed KNN. Set sizes between nodes will be different. \n"
            ),
        )
        edgelist = self.idx_partner_to_edgelist(idx, output_type="cudf")
        G = self.cugraph.from_cudf_edgelist(edgelist)
        # 2. Jaccard
        if two_hop:
            jac_edgelist = self.cugraph.jaccard(
                G
            )  # This will compute two-hop; expensive
        else:
            jac_edgelist = self.cugraph.jaccard(G, vertex_pair=edgelist)
        loguru.logger.info(
            (
                f"Jaccard GPU edgelist constructed in {time.time() - subtic}"
                f"seconds \n"
            ),
        )
        return jac_edgelist


class GraphClustererCPU:
    """Performs graph clustering (or community detection) on the CPU
    (leidenalg)."""

    def edgelist_to_igraph(self, edgelist: pd.DataFrame) -> ig.Graph:
        """Converts an edgelist to an igraph.Graph object.

        Args:
            edgelist: Edgelist with columns denoting the source node,
                destination node, and edge weight.
        """
        if isinstance(edgelist, pd.DataFrame):
            return ig.Graph.DataFrame(edgelist, directed=False)
        else:
            raise ValueError("Unsupported edgelist type.")

    def compute_louvain(
        self,
        igraph: ig.Graph,
        resolution: float,
        max_iter: int = 500,
        min_size: int = 10,
    ) -> tuple[np.ndarray, float]:
        """Perform Louvain community detection on the CPU (leidenalg).

        Args:
            igraph: igraph.Graph object.
            resolution: Resolution parameter for the Louvain algorithm.
            max_iter: Maximum number of iterations.
            min_size: Minimum cluster size. Clusters with less than this number
                of nodes are assigned to -1.

        Returns:
            Tuple of the cluster labels and the quality score.
        """
        raise NotImplementedError()

    def compute_leiden(
        self,
        igraph: ig.Graph,
        resolution: float,
        max_iter: int = -1,
        min_size: int = 10,
    ) -> tuple[np.ndarray, float]:
        """Perform Leiden community detection on the CPU (leidenalg).

        Args:
            igraph: igraph.Graph object.
            resolution: Resolution parameter for the Louvain algorithm.
            max_iter: Maximum number of iterations. If -1, runs until it reaches
                an iteration with no improvement in quality.
            min_size: Minimum cluster size. Clusters with less than this number
                of nodes are assigned to -1.

        Returns:
            Tuple of the cluster labels and the quality score.
        """
        # Leidenalg
        partition = lg.find_partition(
            igraph,
            lg.RBConfigurationVertexPartition,
            resolution_parameter=resolution,
            weights="jaccard_coeff",
            n_iterations=max_iter,
        )
        cdf = np.asarray(partition.membership)
        cdf = self._sort_by_size(cdf, min_size)
        Q = partition.q
        return cdf, Q

    def _sort_by_size(self, clusters: np.ndarray, min_size: int) -> np.ndarray:
        """Relabel clustering in order of descending cluster size.

        New labels are consecutive integers beginning at 0
        Clusters that are smaller than min_size are assigned to -1

        Args:
            clusters: Array of cluster labels.
            min_size: Minimum cluster size.

        Returns:
            Array of cluster labels re-labeled by size.
        """
        relabeled = np.zeros(clusters.shape, dtype=np.int32)
        sizes = [sum(clusters == x) for x in np.unique(clusters)]
        o = np.argsort(sizes)[::-1]
        for i, c in enumerate(o):
            if sizes[c] > min_size:
                relabeled[clusters == c] = i
            else:
                relabeled[clusters == c] = -1
        return relabeled


class GraphClustererGPU:
    """Performs graph clustering (or community detection) on the GPU
    (cugraph)."""

    def __init__(self):
        self.check_if_GPU_version_installed()

    def check_if_GPU_version_installed(self):
        """Check if the necessary RAPIDS (GPU) packages are installed.

        Raises:
            ImportError: If the necessary RAPIDS packages are not installed.
                Provides a message directing the user on how to install the
                missing packages.
        """
        try:
            import cugraph

            self.cugraph = cugraph
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cugraph")) from e

        try:
            import cudf

            self.cudf = cudf
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cudf")) from e

        try:
            import cupy

            self.cupy = cupy
        except ImportError as e:
            raise ImportError(gpu_import_error_message("cupy")) from e

    def edgelist_to_cugraph(
        self,
        edgelist: Union[pd.DataFrame, "cudf.DataFrame"],  # type: ignore # noqa: F821
    ) -> "cugraph.Graph":  # type: ignore # noqa: F821
        """Converts an edgelist to an cugraph.Graph object.

        Args:
            edgelist: Edgelist with columns denoting the source node,
                destination node, and edge weight.

        Returns:
            cugraph.Graph object.

        Raises:
            ValueError: If the edgelist type is not supported.
        """
        G = self.cugraph.Graph()
        if isinstance(edgelist, pd.DataFrame):
            G.from_pandas_edgelist(
                edgelist,
                source="first",
                destination="second",
                weight="jaccard_coeff",
            )
        elif isinstance(edgelist, self.cudf.DataFrame):
            G.from_cudf_edgelist(
                edgelist,
                source="first",
                destination="second",
                edge_attr="jaccard_coeff",
            )
        else:
            raise ValueError("Unsupported edgelist type.")

        return G

    def compute_louvain(
        self,
        cgraph: "cugraph.Graph",  # type: ignore # noqa: F821
        resolution: float,
        max_iter: int = 500,
        min_size: int = 10,
    ) -> tuple["cupy.ndarray", float]:  # type: ignore # noqa: F821
        """Perform Louvain community detection on the GPU (cugraph).

        Args:
            cgraph: cugraph.Graph object.
            resolution: Resolution parameter for the Louvain algorithm.
            max_iter: Maximum number of iterations.
            min_size: Minimum cluster size. Clusters with less than this number
                of nodes are assigned to -1.

        Returns:
            Tuple of the cluster labels and the quality score.
        """
        subtic = time.time()
        loguru.logger.info(
            (
                f"Performing Louvain on GPU: \n"
                f"\t Resolution = {resolution}\n"
                f"\t Max iterations = {max_iter}\n"
                f"\t Min cluster size = {min_size}\n"
            ),
        )
        cdf, Q = self.cugraph.louvain(
            cgraph, resolution=resolution, max_iter=max_iter
        )
        cdf = self._sort_vertex_values(cdf, min_size)
        loguru.logger.info(
            f"cugraph.louvain computed in {time.time() - subtic} seconds \n",
        )
        return cdf, Q

    def compute_leiden(
        self,
        cgraph: "cugraph.Graph",  # type: ignore # noqa: F821
        resolution: float,
        max_iter: int = 500,
        min_size: int = 10,
    ) -> tuple["cupy.ndarray", float]:  # type: ignore # noqa: F821
        """Perform Leiden community detection on the GPU (cugraph).

        Args:
            cgraph: cugraph.Graph object.
            resolution: Resolution parameter for the Leiden algorithm.
            max_iter: Maximum number of iterations.
            min_size: Minimum cluster size. Clusters with less than this number
                of nodes are assigned to -1.

        Returns:
            Tuple of the cluster labels and the quality score
        """
        subtic = time.time()
        loguru.logger.info(
            (
                f"Performing Leiden on GPU: \n"
                f"\t Resolution = {resolution}\n"
                f"\t Max iterations = {max_iter}\n"
                f"\t Min cluster size = {min_size}\n"
            ),
        )
        cdf, Q = self.cugraph.leiden(
            cgraph, resolution=resolution, max_iter=max_iter
        )
        cdf = self._sort_vertex_values(cdf, min_size)
        loguru.logger.info(
            f"cugraph.leiden computed in {time.time() - subtic} seconds \n",
            flush=True,
        )
        loguru.logger.info(f"Q = {Q}", flush=True)
        return cdf, Q

    def _sort_vertex_values(self, cdf, min_size):
        cdf = cdf.sort_values(by="vertex").partition.values
        cdf = self._sort_by_size(cdf, min_size)
        return cdf

    def _sort_by_size(
        self,
        clusters: "cupy.ndarray",  # type: ignore # noqa: F821
        min_size: int,
    ) -> "cupy.ndarray":  # type: ignore # noqa: F821
        """
        Relabel clustering in order of descending cluster size.
        New labels are consecutive integers beginning at 0
        Clusters that are smaller than min_size are assigned to -1.
        Adapted from https://github.com/jacoblevine/PhenoGraph.

        Args:
            clusters: Array of cluster labels.
            min_size: Minimum cluster size.

        Returns:
            Array of cluster labels re-labeled by size.

        """
        relabeled = self.cupy.zeros(clusters.shape, dtype=int)
        _, counts = self.cupy.unique(clusters, return_counts=True)
        # sizes = cp.array([cp.sum(clusters == x) for x in cp.unique(clusters)])
        o = self.cupy.argsort(counts)[::-1]
        for i, c in enumerate(o):
            if counts[c] > min_size:
                relabeled[clusters == c] = i
            else:
                relabeled[clusters == c] = -1
        return relabeled


class HybridPhenographModular:
    """Hybrid implementation of Phenograph with choice of CPU and/or GPU for
    KNN, Jaccard refinement, and graph clustering.

    Args:
        knn: Backend for the KNN search. Options are 'CPU' or 'GPU'.
        refiner: Backend for the Jaccard refinement. Options are 'CPU' or 'GPU'.
        clusterer: Backend for the graph clustering. Options are 'CPU' or 'GPU'.
        clustering: Clustering algorithm to use. Options are 'louvain' or 'leiden'.
    """

    VALID_KNN_BACKENDS = ["CPU", "GPU"]
    VALID_REF_BACKENDS = ["CPU", "GPU"]
    VALID_GCLUST_BACKENDS = ["CPU", "GPU"]  # Must be in cugraph
    VALID_GCLUSTERERS = ["louvain", "leiden"]

    def __init__(
        self,
        knn: Literal["CPU", "GPU"] = "CPU",
        refiner: Literal["CPU", "GPU"] = "CPU",
        clusterer: Literal["CPU", "GPU"] = "CPU",
        clustering: Literal["louvain", "leiden"] = "leiden",
    ):
        self.knn = self.create_knn(knn)
        self.knn_backend = knn
        self.refiner = self.create_refiner(refiner)
        self.refiner_backend = refiner
        self.clusterer = self.create_clusterer(clusterer)
        self.clusterer_backend = clusterer
        self.clustering = clustering

    def create_knn(self, knn: Literal["CPU", "GPU"]) -> KNNCPU | KNNGPU:
        """Create the KNN backend based on the input string.

        Args:
            knn: Backend for the KNN search. Options are 'CPU' or 'GPU'.

        Returns:
            KNN object.
        """
        if knn == "GPU":
            return KNNGPU()
        elif knn == "CPU":
            return KNNCPU()
        else:
            raise TypeError("Unsupported KNN type")

    def create_refiner(
        self, refiner: Literal["CPU", "GPU"]
    ) -> JaccardRefinerCPU | JaccardRefinerGPU:
        """Create the refiner backend based on the input string.

        Args:
            refiner: Backend for the Jaccard refinement. Options are 'CPU' or
                'GPU'.

        Returns:
            JaccardRefiner object.
        """
        if refiner == "CPU":
            return JaccardRefinerCPU()
        elif refiner == "GPU":
            return JaccardRefinerGPU()
        else:
            raise TypeError("Unsupported refiner type")

    def create_clusterer(
        self, clusterer: Literal["CPU", "GPU"]
    ) -> GraphClustererCPU | GraphClustererGPU:
        """Create the clusterer backend based on the input string.

        Args:
            clusterer: Backend for the graph clustering. Options are 'CPU' or
                'GPU'.

        Returns:
            GraphClusterer object.
        """
        if clusterer == "CPU":
            return GraphClustererCPU()
        elif clusterer == "GPU":
            return GraphClustererGPU()
        else:
            raise TypeError("Unsupported clustering type")

    def knn_func(
        self,
        data,
        n_neighbors,
        algorithm,
        metric,
        p,
        output_type,
        n_jobs,
        two_pass_precision,
    ):
        """Calls the KNN function based on the backend. Parses the parameters
        accordingly.

        TODO: implement as a dispatch
        """
        if self.knn_backend == "CPU":
            return self.knn.compute_neighbors(
                data, n_neighbors, algorithm, metric, p, n_jobs
            )  # d, idx

        elif self.knn_backend == "GPU":
            return self.knn.compute_neighbors(
                data,
                n_neighbors,
                algorithm,
                metric,
                p,
                output_type=output_type,
                two_pass_precision=two_pass_precision,
            )  # Movement to cpu handled in refiner func checks.

        else:
            raise TypeError("Unsupported KNN type")

    def refiner_func(self, idx):
        """Calls the refiner function based on the backend. Parses the parameters
        accordingly.

        TODO: implement as a dispatch
        """
        if (
            self.refiner_backend == "CPU" and self.knn_backend == "GPU"
        ):  # IF GPU -> CPU
            return self.refiner.compute_jaccard(idx.get())
        else:  # Within GPU
            return self.refiner.compute_jaccard(idx)  # jaccard edgelist

    def cluster_func(self, edgelist, resolution, min_size):
        """Calls the graph clusterer based on the backend. Parses the parameters
        accordingly.

        TODO: implement as a dispatch
        """
        if self.clusterer_backend == "CPU":
            graph = self.clusterer.edgelist_to_igraph(
                edgelist=edgelist
            )  # leidenalg

        elif self.clusterer_backend == "GPU":
            graph = self.clusterer.edgelist_to_cugraph(
                edgelist=edgelist
            )  # cugraph

        else:
            raise TypeError("Unsupported clusterer")

        # TODO: Run isolated node checks; latest branch includes isolated nodes in graph structure -> 24.04.6

        if self.clustering == "louvain":
            return self.clusterer.compute_louvain(
                graph, resolution=resolution, min_size=min_size
            )
        elif self.clustering == "leiden":
            return self.clusterer.compute_leiden(
                graph, resolution=resolution, min_size=min_size
            )
        else:
            raise TypeError("Unsupported clustering type")

    def cluster(
        self,
        adata: AnnData,
        n_neighbors: int,
        algorithm: str = "auto",
        metric: str = "euclidean",
        p: int = 2,
        n_jobs: int = -1,
        two_pass_precision: bool = False,
        resolution: float = 1.0,
        min_size: int = 10,
    ) -> tuple[Union[np.ndarray, "cupy.ndarray"], float]:  # type: ignore # noqa: F821
        """Run the Phenograph algorithm depending on the backends chosen for
        each step.

        Args:
            adata: AnnData object.
            n_neighbors: Number of nearest neighbors to compute for the KNN
                graph.
            algorithm: Algorithm to use for nearest neighbors search.
            metric: Distance metric to use for the nearest neighbors search.
            p: Power parameter for the Minkowski metric.
            n_jobs: Number of parallel jobs to run. (CPU only)
            two_pass_precision: Use two-pass algorithm for precision. (GPU only)
            resolution: Resolution parameter for graph clustering.
            min_size: Minimum cluster size. Clusters with less than this number
                of nodes are assigned to -1.
        """
        # Read in X_pca data
        embedding_name = "X_pca_harmony"
        if embedding_name not in adata.obsm:
            if "X_pca" not in adata.obsm:
                raise ValueError("No PCA embedding found in AnnData.obsm")
            embedding_name = "X_pca"
        data = adata.obsm[embedding_name]

        # KNN
        d, idx = self.knn_func(
            data,
            n_neighbors,
            algorithm,
            metric,
            p,
            "cupy",
            n_jobs,
            two_pass_precision,
        )  #

        # Jaccard
        refined_edgelist = self.refiner_func(idx)
        # Graph Clustering
        cdf, Q = self.cluster_func(
            refined_edgelist, resolution=resolution, min_size=min_size
        )
        return cdf, Q


class HybridPhenographSearch(HybridPhenographModular):
    """Performs a parameter search over a range of Ks and Rs for the Phenograph
    algorithm."""

    def __init__(
        self,
        knn: Literal["CPU", "GPU"] = "GPU",
        refiner: Literal["CPU", "GPU"] = "CPU",
        clusterer: Literal["CPU", "GPU"] = "GPU",
        clustering: Literal["louvain", "leiden"] = "leiden",
    ):
        super().__init__(knn, refiner, clusterer, clustering)

        #: Stores the `k` and `r` parameters in a grid format
        self.param_grid = {}

        #: Stores the clustering results for each `k` and `r` combination
        self.data_grid = {}

        #: Stores the quality score for each `k` and `r` combination
        self.quality_grid = {}

        # Enforce backends so data structures are compatible
        if self.clusterer_backend == "GPU":
            self._set_array_backend("GPU")
            self._set_df_backend("GPU")
        else:
            self._set_array_backend("CPU")
            self._set_df_backend("CPU")

    def _set_array_backend(self, backend):
        """Sets the dataframe backend for processing the clustering outputs.
        If clusterer is GPU, will produce cudf outputs, so backend is set to
        cudf. Otherwise, pandas is used."""
        if backend == "GPU":
            try:
                import cupy

                self.xp = cupy
            except ImportError as e:
                raise ImportError(gpu_import_error_message("cupy")) from e

        elif backend == "CPU":
            try:
                import numpy

                self.xp = numpy
            except ImportError as e:
                raise ImportError("numpy not installed.") from e

        else:
            raise TypeError("Unsupported backend")

    def _set_df_backend(self, backend):
        """Sets the dataframe backend for processing the clustering outputs.
        If clusterer is GPU, will produce cudf outputs, so backend is set to
        cudf. Otherwise, pandas is used."""
        if backend == "GPU":
            try:
                import cudf as cudf

                self.df = cudf
            except ImportError as e:
                raise ImportError(gpu_import_error_message("cudf")) from e

        elif backend == "CPU":
            try:
                import pandas as pd

                self.df = pd
            except ImportError as e:
                raise ImportError("pandas not installed.") from e

        else:
            raise TypeError("Unsupported clusterer")

    def _log_current_time(self):
        if self.log_time:
            loguru.logger.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]"
            )

    def parameter_search(
        self,
        adata: AnnData,
        ks: list[int],
        embedding_name: str = "X_pca_harmony",
        algorithm: str = "brute",
        metric: str = "euclidean",
        p: int = 2,
        n_jobs: int = -1,
        output_type: str = "cupy",
        two_pass_precision: bool = False,
        rs: list[float] | None = None,
        min_size: int = 10,
        save: bool = False,
        save_name: str = "data",
        enable_cold_start: bool = True,  # NOT IMP
        cold_from: str | None = None,
        log_time: bool = True,
    ) -> AnnData:
        """Extension of `self.cluster` performed over many Ks and Rs.

        Args:
            adata: AnnData object.
            ks: List of K values to search over.
            embedding_name: Name of the embedding in `adata.obsm` to use.
            algorithm: Algorithm to use for nearest neighbors search.
            metric: Distance metric to use for the nearest neighbors search.
            p: Power parameter for the Minkowski metric.
            n_jobs: Number of parallel jobs to run. (CPU only)
            output_type: Output type for the KNN search. Options are 'cupy' or
                'cudf'.
            two_pass_precision: Use two-pass algorithm for precision. (GPU only)
            rs: List of R values to search over.
            min_size: Minimum cluster size. Clusters with less than this number
                of nodes are assigned to -1.
            save: Save the results to disk.
            save_name: Name of the file to save the results to.
            enable_cold_start: (NOT IMPLEMENTED) Enable cold start to resume
                from a previous run.
            cold_from: File to resume from if cold start is enabled.
            log_time: Log the time at each step of the process.

        Returns:
            AnnData object with the following stored:
            1) Clustering results stored as pd.DataFrames in
                `adata.obsm[*_labels]`, where the columns represent a single
                clustering run.
            2) Graph clustering quality scores stored in
                `adata.uns[*_quality_scores]`.
        """
        if rs is None:
            rs = [1.0]
        self.log_time = log_time
        loguru.logger.info("Beginning Parameter Search... \n")
        self._log_current_time()

        # Use PCA embeddings as input data
        if embedding_name not in adata.obsm:
            if "X_pca" not in adata.obsm:
                raise ValueError("No PCA embedding found in AnnData.obsm")
            embedding_name = "X_pca"
        data = adata.obsm[embedding_name]

        # Set up param grid
        self.param_grid["ks"] = ks
        self.param_grid["rs"] = rs
        self.param_grid["min_size"] = min_size
        # Annotate Adata with parameters
        adata.uns["param_grid"] = self.param_grid

        # Only do KNN once
        max_k = max(ks)
        d, idx = self.knn_func(
            data,
            n_neighbors=max_k,
            algorithm=algorithm,
            metric=metric,
            p=p,
            output_type=output_type,
            n_jobs=n_jobs,
            two_pass_precision=two_pass_precision,
        )

        self._log_current_time()
        knn_name = (
            f"indices_Kmax{max_k}_alg{algorithm}_metric{metric}_p{p}.npy"
        )

        if save:
            # uncompressed for fast read write
            self.xp.save(f"{save_name}/{knn_name}", idx)

        for k in ks:
            idx_subset = idx[:, :k]
            refined_edgelist = self.refiner_func(idx_subset)
            self._log_current_time()

            if save:
                jaccard_name = f"edgelist_jaccard_K{k}.feather"
                refined_edgelist.to_feather(f"{save_name}/{jaccard_name}")

            # Then for that Jaccard Graph, cluster for each resolution R
            for r in rs:
                clusters, Q = self.cluster_func(
                    refined_edgelist, resolution=r, min_size=min_size
                )
                self._log_current_time()

                # cache data
                clusters = clusters.tolist()  # to cpu memory as py lists

                self.data_grid[(k, r)] = (
                    clusters  # Back to memory if gpu rgardles
                )
                self.quality_grid[(k, r)] = Q

                if save:
                    clusters_save = clusters.copy()
                    clusters_save.append(Q)
                    clustering_name = f"clusters_K{k}_r{r}.npy"  # TODO: Ideally savez as two objects; Clusters and Q, rather than having Q in clusters array;
                    self.xp.save(
                        f"{save_name}/{clustering_name}", clusters_save
                    )
                    del clusters_save
                    gc.collect()

            if save:
                del idx_subset
                del refined_edgelist
                gc.collect()

        # Create an index mapping of range indices to the actual indices
        number_index = [str(x) for x in range(adata.shape[0])]
        self.index_map = dict(zip(number_index, adata.obs.index, strict=False))
        results = self.df.DataFrame(self.data_grid)
        results.index = results.index.astype(str)
        if self.clusterer_backend == "GPU":
            # cudf Index lacks map
            reindexed = results.index.to_series().map(self.index_map)
        else:
            reindexed = results.index.map(self.index_map)
        results.index = reindexed

        adata = self._label_adata(adata, results)
        return adata

    def _label_adata(self, adata, results_df):
        # Store labels in an obsm matrix
        OBSM_ADDED_KEY = self.__class__.__name__ + "_labels"

        # Store what parameters each column label represents in the obsm matrix
        # UNS_ADDED_KEY_LABELMAP = self.__class__.__name__ + "_label_map"
        # label_map = {k:v for k,v in enumerate(self.data_grid.keys())} # obsm index : (k,r)
        # results_df.columns = results_df.columns.map(label_map)
        # Convert tuple to string so its Zarr writable
        # Just go with "K_R" for now
        if self.clusterer_backend == "GPU":
            results_df = results_df.to_pandas()
        results_df.columns = [f"{k}_{r}" for k, r in results_df.columns]
        adata.obsm[OBSM_ADDED_KEY] = results_df
        # adata.uns[UNS_ADDED_KEY_LABELMAP] = label_map

        # Add param grids as quality grid
        UNS_ADDED_KEY_QUALITYSCORE = (
            self.__class__.__name__ + "_quality_scores"
        )
        self._set_df_backend("CPU")  # pandas
        df = self.df.DataFrame(
            self.df.DataFrame(
                self.quality_grid, index=list(range(len(self.quality_grid)))
            ).T[0]
        )
        df = df.reset_index()
        df = df.rename(columns={"level_0": "K", "level_1": "R"})
        df = df.rename(columns={0: "modularity_score"})

        adata.uns[UNS_ADDED_KEY_QUALITYSCORE] = df

        return adata


class ScanpyClustering:
    """Performs the Scanpy clustering workflow on the CPU (scanpy) or on
    the GPU (rapids-singlecell).
    """

    MAX_GPU_LEIDEN_ITER = 500
    VALID_BACKENDS = ["CPU", "GPU"]

    def __init__(self, backend: Literal["CPU", "GPU"] = "CPU") -> None:
        if backend not in self.VALID_BACKENDS:
            raise ValueError(f"Backend must be one of {self.VALID_BACKENDS}.")
        self.backend = backend
        self._set_backend_sc(backend)
        self._set_array_backend(backend)
        self._set_df_backend(backend)

    def _set_backend_sc(self, backend):
        if backend == "GPU":
            try:
                import rapids_singlecell as sc

                self.sc = sc
            except ImportError as e:
                raise ImportError(
                    gpu_import_error_message("rapids-singlecell")
                ) from e
        elif backend == "CPU":
            import scanpy as sc

            self.sc = sc
        else:
            raise ValueError("Backend must be either 'CPU' or 'GPU'.")

    def _set_array_backend(self, backend):
        """Sets the dataframe backend for processing the clustering outputs.
        If clusterer is GPU, will produce cudf outputs, so backend is set to
        cudf. Otherwise, pandas is used."""
        if backend == "GPU":
            try:
                import cupy

                self.xp = cupy
            except ImportError as e:
                raise ImportError(gpu_import_error_message("cupy")) from e

        elif backend == "CPU":
            try:
                import numpy

                self.xp = numpy
            except ImportError as e:
                raise ImportError("pandas not installed.") from e

        else:
            raise TypeError("Unsupported clusterer")

    def _set_df_backend(self, backend):
        """Sets the dataframe backend for processing the clustering outputs.
        If clusterer is GPU, will produce cudf outputs, so backend is set to
        cudf. Otherwise, pandas is used."""
        if backend == "GPU":
            try:
                import cudf as cudf

                self.df = cudf
            except ImportError as e:
                raise ImportError(gpu_import_error_message("cudf")) from e

        elif backend == "CPU":
            try:
                import pandas as pd

                self.df = pd
            except ImportError as e:
                raise ImportError("pandas not installed.") from e

        else:
            raise TypeError("Unsupported clusterer")

    def graph_cluster(
        self,
        adata: AnnData,
        resolution: float,
        random_state: int = 0,
        n_iter: int = -1,
    ) -> tuple[np.ndarray, float]:
        """Perform the leiden graph clustering on the connectivities graph
        generated by scanpy.pp.neighbors (or rapids_singlecell.pp.neighbors).

        Args:
            adata: AnnData object.
            resolution: Resolution parameter for the leiden algorithm.
            random_state: Random seed for the algorithm.
            n_iter: Maximum number of iterations. If -1, runs until it reaches
                an iteration with no improvement in quality.

        Returns:
            Tuple of the cluster labels and the quality score.
        """
        if self.backend == "CPU":
            # from scanpy src
            import leidenalg as la
            from scanpy._utils import get_igraph_from_adjacency as create_graph

            ig = create_graph(adata.obsp["connectivities"])
            w = np.array(ig.es["weight"]).astype(np.float64)
            part = la.find_partition(
                ig,
                la.RBConfigurationVertexPartition,
                weights=w,
                seed=random_state,
                resolution_parameter=resolution,
                n_iterations=n_iter,
            )
            groups = np.array(part.membership)
            Q = part.modularity
            return groups, Q

        elif self.backend == "GPU":
            # from rapids_singlecell src
            from cugraph import leiden as culeiden
            from rapids_singlecell.tools import _create_graph as create_graph

            cg = create_graph(adata.obsp["connectivities"])
            if n_iter == -1:
                n_iter = self.MAX_GPU_LEIDEN_ITER  # enforce max iters for gpu

            leiden_parts, Q = culeiden(
                cg,
                resolution=resolution,
                random_state=random_state,
                max_iter=n_iter,
            )

            # Format output
            groups = (
                leiden_parts.to_pandas()
                .sort_values("vertex")[["partition"]]
                .to_numpy()
                .ravel()
            )
            return groups, Q

        else:
            raise ValueError("Backend must be either 'CPU' or 'GPU'.")


class ScanpyClusteringSearch(ScanpyClustering):
    """Performs a parameter search over a range of Ks and Rs for the Scanpy
    clustering workflow."""

    def __init__(self, backend="CPU"):
        super().__init__(backend)

        #: Stores the `k` and `r` parameters in a grid format
        self.param_grid = {}

        #: Stores the clustering results for each `k` and `r` combination
        self.data_grid = {}

        #: Stores the quality score for each `k` and `r` combination
        self.quality_grid = {}

    def _log_current_time(self, message):
        if self.log_time:
            loguru.logger.info(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n{message}",
            )

    def parameter_search(
        self,
        adata: AnnData,
        ks: list[int],
        embedding_name: str = "X_pca_harmony",
        n_pcs: int | None = None,
        rs: list[float] | None = None,
        random_state: int = 0,
        n_iter: int = -1,
        min_size: int = 10,
        log_time: bool = True,
    ):
        """Perform parameter search for scanpy clustering workflow.

        Args:
            adata: AnnData object.
            ks: List of K values to search over.
            embedding_name: Name of the embedding in `adata.obsm` to use.
            n_pcs: Number of principal components to use for PCA.
            rs: List of R values to search over.
            random_state: Random seed for the algorithm.
            n_iter: Maximum number of iterations. If -1, runs until it reaches
                an iteration with no improvement in quality.
            min_size: Minimum cluster size. Clusters with less than this number
                of nodes are assigned to -1.
            log_time: Log the time at each step of the process.

        Returns:
            AnnData object with the following stored:
            1) Clustering results stored as pd.DataFrames in
                `adata.obsm[*_labels]`, where the columns represent a single
                clustering run.
            2) Graph clustering quality scores stored in
                `adata.uns[*_quality_scores]`.
        """
        if rs is None:
            rs = [1.0]

        self.log_time = log_time
        self._log_current_time("Starting parameter search")

        # Use PCA embeddings as input data
        if embedding_name not in adata.obsm:
            if "X_pca" not in adata.obsm:
                self.sc.pp.pca(adata, n_comps=n_pcs)
            embedding_name = "X_pca"
        # data = adata.obsm[embedding_name]

        # Set up param grid
        self.param_grid["ks"] = ks
        self.param_grid["rs"] = rs

        adata.uns["param_grid"] = self.param_grid

        # Knn / neighbors
        for k in ks:
            self.sc.pp.neighbors(
                adata, n_neighbors=10, use_rep=embedding_name
            )  # X_pca or X_pca_harmony
            self._log_current_time(f"Finished KNN with k={k}")

            for r in rs:
                labels, Q = self.graph_cluster(
                    adata,
                    resolution=r,
                    random_state=random_state,
                    n_iter=n_iter,
                )

                # TODO -> min_size filtering
                # self.sc.tl.leiden(adata, resolution=r)
                self._log_current_time(
                    f"\tFinished Leiden with resolution={r}"
                )

                # Cache data
                clusters = labels.tolist()
                self.data_grid[(k, r)] = clusters
                self.quality_grid[(k, r)] = Q

        # Create an index mapping of range indices to the actual indices
        number_index = [str(x) for x in range(adata.shape[0])]
        self.index_map = dict(zip(number_index, adata.obs.index, strict=False))
        results = self.df.DataFrame(self.data_grid)
        results.index = results.index.astype(str)
        results.index = results.index.map(self.index_map)  # has no attr map?

        adata = self._label_adata(adata, results)
        return adata

    def _label_adata(self, adata, results_df):
        # Store labels in an obsm matrix
        OBSM_ADDED_KEY = self.__class__.__name__ + "_labels"

        # Store what parameters each column label represents in the obsm matrix
        # UNS_ADDED_KEY_LABELMAP = self.__class__.__name__ + "_label_map"
        # label_map = {k:v for k,v in enumerate(self.data_grid.keys())} # obsm index : (k,r)
        # results_df.columns = results_df.columns.map(label_map)
        # Convert tuple to string so its Zarr writable
        # Just go with "K_R" for now
        if self.clusterer_backend == "GPU":
            results_df = results_df.to_pandas()
        results_df.columns = [f"{k}_{r}" for k, r in results_df.columns]
        adata.obsm[OBSM_ADDED_KEY] = results_df
        # adata.uns[UNS_ADDED_KEY_LABELMAP] = label_map

        # Add param grids as quality grid
        UNS_ADDED_KEY_QUALITYSCORE = (
            self.__class__.__name__ + "_quality_scores"
        )
        self._set_df_backend("CPU")  # pandas
        df = self.df.DataFrame(
            self.df.DataFrame(
                self.quality_grid, index=list(range(len(self.quality_grid)))
            ).T[0]
        )
        df = df.reset_index()
        df = df.rename(columns={"level_0": "K", "level_1": "R"})
        df = df.rename(columns={0: "modularity_score"})

        adata.uns[UNS_ADDED_KEY_QUALITYSCORE] = df

        return adata


def cluster(
    adata: AnnData,
    recipe: Literal["phenograph", "scanpy"],
    backend: Literal["CPU", "GPU"] = "CPU",
    **kwargs,
) -> AnnData:
    pass

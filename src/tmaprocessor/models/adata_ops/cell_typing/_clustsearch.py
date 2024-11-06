""" Originally from phenotyping.py """
import os
import gc
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import adjusted_rand_score
from datetime import datetime

""" Originally from phenotyping.py """
from ..._utils import overrides
import time
import sklearn.neighbors as skn
import numpy as np
from types import MappingProxyType

import pandas as pd
# import networkx as nx
# import igraph as ig
# import leidenalg as lg

from scipy import sparse as sp

# TODO: make this cpu only 
from phenograph.core import (
    parallel_jaccard_kernel
)
import igraph as ig
import leidenalg as lg

class KNN():
    def compute_neighbors():# data, K, output_type
        raise NotImplementedError("Abstract method.")

    def indices_to_edgelist():
        """ Convert indices output from sklearn-like neighbours.kneighbors outputs. """
        raise NotImplementedError("Abstract method.")

    def edgelist_to_graph():
        """ Convert edgelist to graph. """
        raise NotImplementedError("Abstract method.")
    
    def _remove_self_distances(self, d, idx):
        raise NotImplementedError("Abstract method.")
    
    def _log_equivalent_metrics(self, metric, p):
        if metric == "minkowski" and p == 2:
            log_metric = "euclidean"
        elif metric == "minkowski" and p == 1:
            log_metric = "manhattan"
        else:
            log_metric = None 
        return metric, log_metric


class KNNCPU(KNN):
    @overrides(KNN)
    def _remove_self_distances(self, d, idx):
        """ Remove self distances from the nearest neighbors output. """
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
        data, 
        n_neighbors, 
        algorithm="auto", # KNN search method
        metric="minkowski", 
        p=2, # Power parameter for Minkowski metric
        n_jobs=-1):
        """ From phenograph.core.py """
        # time operation
        subtic = time.time()
        
        # Log equivalent distance metric for clarity
        metric, log_metric = self._log_equivalent_metrics(metric, p)
        k_string = f"\t K = {n_neighbors}\n"
        distance_string = f"\t Distance Metric = {metric}\n" if log_metric is None \
            else f"\t Distance Metric = {log_metric} ({metric} backend; SLOWER)\n"
        algorithm_string = f"\t Search Algorithm = {algorithm}\n"

        print(f"Performing KNN search on CPU: \n" + \
              k_string + \
              distance_string + \
              algorithm_string,
              flush = True)

        # Enforce brute force if metric is cosine or correlation
        if metric in ["cosine", "correlation"]:
            algorithm = "brute"
            print(
                f"Enforcing brute search for {metric} metric", 
                flush=True)

        knn = skn.NearestNeighbors(
            n_neighbors=n_neighbors + 1, # Due to results including self
            algorithm=algorithm, 
            metric=metric, 
            p=p, 
            n_jobs=n_jobs)
        
        knn.fit(data)
        d, idx = knn.kneighbors(data)
        d, idx = self._remove_self_distances(d, idx)
        print(f"KNN CPU computed in {time.time() - subtic} seconds \n", flush=True)
        return d, idx

class KNNGPU(KNN):
    AVAIL_DIST_METRICS = [
         "euclidean", "l2", "sqeuclidean", "cityblock", "l1", "manhattan",
         "taxicab", "braycurtis", "canberra", "minkowski", "lp", "chebyshev",
         "linf", "jensenshannon", "cosine", "correlation", "inner_product",
         "jaccard", "hellinger", "haversine"]
    
    def __init__(self):
        self.check_if_GPU_version_installed()

    def check_if_GPU_version_installed(self):
        try:
            import cuml
            self.cuml = cuml
        except ImportError:
            raise ImportError("cuml not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")
        
        try:
            import cupy
            self.cupy = cupy
        except ImportError:
            raise ImportError("cupy not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")

    @overrides(KNN)
    def _remove_self_distances(self, d, idx):
        """ Remove self distances from the nearest neighbors output. """
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
        data, 
        n_neighbors, 
        algorithm="auto", # KNN search method
        metric="euclidean", # [‘l1, ‘cityblock’, ‘taxicab’, ‘manhattan’, ‘euclidean’, ‘l2’, ‘braycurtis’, ‘canberra’, ‘minkowski’, ‘chebyshev’, ‘jensenshannon’, ‘cosine’, ‘correlation’]
        p=2, # Power parameter for Minkowski metric
        output_type="cupy", # {‘input’, ‘array’, ‘dataframe’, ‘series’, ‘df_obj’, ‘numba’, ‘cupy’, ‘numpy’, ‘cudf’, ‘pandas’}
        two_pass_precision=True,
        ): 
        """ 
        FuseSOM paper: Cosine/Spearman/Pearson 
        """
        # time operation
        subtic = time.time()
        
        # Log equivalent distance metric for clarity
        metric, log_metric = self._log_equivalent_metrics(metric, p)
        k_string = f"\t K = {n_neighbors}\n"
        distance_string = f"\t Distance Metric = {metric}\n" if log_metric is None \
            else f"\t Distance Metric = {log_metric} ({metric} backend; SLOWER)\n"
        algorithm_string = f"\t Search Algorithm = {algorithm}\n"

        print(f"Performing KNN search on GPU: \n" + \
              k_string + \
              distance_string + \
              algorithm_string,
              flush = True)
        
        # Enforce brute force if metric is cosine or correlation
        if metric in ["cosine", "correlation"]:
            algorithm = "brute"
            print(
                f"Enforcing brute search for {metric} metric", 
                flush=True)
            
        knn = self.cuml.neighbors.NearestNeighbors(
            n_neighbors=n_neighbors + 1, # Due to results including self
            algorithm=algorithm, 
            metric=metric, 
            p=2, 
            output_type=output_type
            )
        # Can technically do rsc.pp.neighbors;
        # rsc.pp.neighbors; if brute -> cupy, returns idx, d 
        # idx, d --> distances as csr_matrix
        # Cast data to cupy array
        X_cupy = self.cupy.asarray(data)
        knn.fit(X_cupy)
        d, idx = knn.kneighbors(X_cupy, two_pass_precision=two_pass_precision)
        d, idx = self._remove_self_distances(d, idx)
        print(f"KNN GPU computed in {time.time() - subtic} seconds \n", flush=True)
        return d, idx
    
class KNNRSC(KNNGPU):
    def __init__(self):
        self.check_if_GPU_version_installed()

    @overrides(KNNGPU)
    def check_if_GPU_version_installed(self):
        try:
            import rapids_singlecell
            self.rsc = rapids_singlecell
        except ImportError:
            raise ImportError("rapids-singlecell not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")

    def _reverse_csr_matrix(self, csr_matrix):
        """ Undo the CSR operation of a cupyx sparse csr matrix.
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
        adata,
        n_neighbors,
        n_pcs,
        use_rep,
        random_state=0,
        algorithm="auto",
        metric="euclidean",
        metric_kwds=MappingProxyType({}),
        key_added="rsc_neighbors",
        output_type="csr"):
        """ Let rsc.pp.neighbors handle the backends;

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
            copy=False)
        
        weighted_adjacency_csr = adata.obsp[key_added + "_distances"] # cupyx.scipy.sparse.csr_matrix 

        if output_type == "csr":
            return weighted_adjacency_csr
        else:
            # Return to d, idx format with no self distances in the KNN graph; 
            d, idx = self._reverse_csr_matrix(weighted_adjacency_csr)
            d, idx = self._remove_self_distances(d, idx)
            return d, idx

""" Originally from phenotyping.py """

"""
TODO: Maybe extend link prediction from CuGraph to swap out coefficients;

Jaccard vs Overlap:
- Jaccard: O(|A|+|B|), but since K = |A| = |B|, then O(N), across all pairs
Same for Overlap

Issue with GPU Jaccard is the large memory demand. 
https://synergy.cs.vt.edu/pubs/papers/sathre-fpga-jaccard-hpec2022.pdf
https://www.researchgate.net/publication/285835027_GPUSCAN_GPU-based_Parallel_Structural_Clustering_Algorithm_for_Networks
https://www.nature.com/articles/s43588-023-00465-8

Memory CuGraph resolve: https://medium.com/rapids-ai/tackling-large-graphs-with-rapids-cugraph-and-unified-virtual-memory-b5b69a065d4
try with JaccardGPU;
rmm.reinitialize(managed_memory=True)
assert(rmm.is_initialized())

"""

class GraphRefiner():
    def compute_jaccard():
        raise NotImplementedError("Abstract method.")
    
    def idx_partner_to_edgelist(self, idx):
        """ Converts the KNN idx outputs to an edgelist format:
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
        """
        raise NotImplementedError("Abstract method.")

    def idx_partner_to_coo(self, idx):
        """ Converts the KNN idx outputs to an coo format:
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
    def coo_symmatrix_to_edgelist(self, coo):
        edgelist = np.vstack(coo.nonzero()).T.tolist()
        # For now pd dataframe struct
        edgelist = pd.DataFrame(edgelist)
        edgelist.columns = ["first", "second"] # Follow convention of jaccard gpu
        edgelist["jaccard_coeff"] = coo.data
        return edgelist
    
    @overrides(GraphRefiner)
    def idx_partner_to_edgelist(self, idx):
        return np.column_stack(
            (np.repeat(np.arange(idx.shape[0]), idx.shape[1]), 
             idx.ravel()))

    @overrides(GraphRefiner)
    def compute_jaccard(self, idx):
        subtic = time.time()
        print(f"Performing Jaccard on CPU: \n" \
              f"\t KNN graph nodes = {idx.shape[0]}\n" \
              f"\t KNN graph K-neighbors = {idx.shape[1]}\n", 
              flush=True)
        # NetworkX-like format
        # NOTE: Below is a direct-neighbor comparison -> one-hop neighbors
        i, j, s = parallel_jaccard_kernel(idx)
        jaccard_graph = sp.coo_matrix(
            (s, (i, j)), shape=(idx.shape[0], idx.shape[0])
            )
        # Graph is un-directed, so we need to symmetrize;
        jaccard_graph = (jaccard_graph + jaccard_graph.transpose()).multiply(0.5)
        # retain lower triangle (for efficiency) ---> Assumes symmetry for KNN graph ..?
        jaccard_coo_symmatrix = sp.tril(jaccard_graph, -1)
        jaccard_edgelist = self.coo_symmatrix_to_edgelist(jaccard_coo_symmatrix)

        jaccard_edgelist = self.add_isolated_nodes(idx.shape[0], jaccard_edgelist)
        
        print(
            f"Jaccard CPU edgelist constructed in {time.time() - subtic}" \
            f"seconds \n", 
            flush=True)
        return jaccard_edgelist

    def add_isolated_nodes(self, n_cells, edgelist):
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
        result = pd.concat([edgelist, isolated_nodes_df], axis=0, ignore_index=True)
        return result

class JaccardRefinerGPU(GraphRefiner):
    def __init__(self):
        self.check_if_GPU_version_installed()

    @overrides(GraphRefiner)
    def idx_partner_to_edgelist(self, idx, output_type="cupy"):
        if isinstance(idx, self.cupy.ndarray):
            edgelist = self.cupy.column_stack(
                    (self.cupy.repeat(self.cupy.arange(idx.shape[0]), idx.shape[1]), 
                    idx.ravel()))
        else:
            raise ValueError("Unsupported idx type. Takes in cupy.ndarray")
        
        if output_type == "cupy":
            return edgelist
            
        elif output_type == "cudf":
            cdf = self.cudf.DataFrame(edgelist).rename(columns={0:"source", 1:"destination"})
            return cdf
        else:
            raise ValueError("Unsupported type")

    def check_if_GPU_version_installed(self):
        try:
            import cugraph
            self.cugraph = cugraph
        except ImportError:
            raise ImportError("cugraph not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")
        
        try:
            import cudf
            self.cudf = cudf
        except ImportError:
            raise ImportError("cudf not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")

        try:
            import rmm
            self.rmm = rmm
        except ImportError:
            raise ImportError("rmm not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")
        
        try:
            import cupy
            self.cupy = cupy
        except ImportError:
            raise ImportError("cupy not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")
        
    def initialise_uvm(self):
        """ Set RMM to allocate all memory as managed memory; unified virtual memory
            aka GPU + CPU """
        print("Initialising rapids memory manager to enable memory oversubscription with unified virtual memory...", flush=True)
        self.rmm.mr.set_current_device_resource(self.rmm.mr.ManagedMemoryResource())
        assert(self.rmm.is_initialized())


    def compute_jaccard(self, idx, two_hop=False):
        self.initialise_uvm()

        subtic = time.time()
        print(f"Performing Jaccard on GPU: \n" \
              f"\t KNN graph nodes = {idx.shape[0]}\n" \
              f"\t KNN graph K-neighbors = {idx.shape[1]}\n", 
              flush=True)
        print(f"NOTE: This performs undirected KNN Jaccard refinement. "\
              f"Different to the original CPU implementation, which is a directed KNN. " \
              f"Set sizes between nodes will be different. \n",
              flush=True)
        edgelist = self.idx_partner_to_edgelist(idx, output_type="cudf")
        G = self.cugraph.from_cudf_edgelist(edgelist)
        # 2. Jaccard
        if two_hop:
            jac_edgelist = self.cugraph.jaccard(G) # This will compute two-hop; expensive
        else:
            jac_edgelist = self.cugraph.jaccard(G, vertex_pair=edgelist)
        print(
            f"Jaccard GPU edgelist constructed in {time.time() - subtic}" \
            f"seconds \n", 
            flush=True)
        return jac_edgelist

""" Originally from phenotyping.py """


class GraphClustererCPU():

    def _igraph_to_networkx(self):
        pass

    def _networkx_to_igraph(self):
        pass

    def edgelist_to_igraph(self, edgelist):
        if isinstance(edgelist, pd.DataFrame):
            return ig.Graph.DataFrame(edgelist, directed=False)
        elif isinstance(edgelist, nx.Graph):    
            return ig.Graph.from_networkx(edgelist)
        else:
            raise ValueError("Unsupported edgelist type.")
        
    def compute_louvain(self, cgraph, resolution, max_iter=500, min_size=10):
        raise NotImplementedError()

    def compute_leiden(self, igraph, resolution, max_iter=-1, min_size=10):
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
    
    def _sort_by_size(self, clusters, min_size):
        """
        Relabel clustering in order of descending cluster size.
        New labels are consecutive integers beginning at 0
        Clusters that are smaller than min_size are assigned to -1
        :param clusters:
        :param min_size:
        :return: relabeled
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

class GraphClustererGPU():
    """ aka CommunityDetector."""
    def __init__(self):
        self.check_if_GPU_version_installed()
        
    def check_if_GPU_version_installed(self):
        try:
            import cugraph
            self.cugraph = cugraph
        except ImportError:
            raise ImportError("cugraph not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")
        
        try:
            import cudf
            self.cudf = cudf
        except ImportError:
            raise ImportError("cudf not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")

        try:
            import cupy
            self.cupy = cupy
        except ImportError:
            raise ImportError("cupy not installed. \
                              Need to install gpu extras of this package: \
                              `pip install 'pcf_analysis_poetry[gpu]'`")
        
    def edgelist_to_cugraph(self, edgelist):
        G = self.cugraph.Graph()
        if isinstance(edgelist, pd.DataFrame):
            G.from_pandas_edgelist(edgelist, source="first", destination="second", weight="jaccard_coeff")
        elif isinstance(edgelist, self.cudf.DataFrame):
            G.from_cudf_edgelist(edgelist, source="first", destination="second", edge_attr="jaccard_coeff")
        else:
            raise ValueError("Unsupported edgelist type.")
        return G
        
    def compute_louvain(self, cgraph, resolution, max_iter=500, min_size=10):
        subtic = time.time()
        print(f"Performing Louvain on GPU: \n" \
              f"\t Resolution = {resolution}\n"
              f"\t Max iterations = {max_iter}\n"
              f"\t Min cluster size = {min_size}\n",
              flush=True)
        cdf, Q = self.cugraph.louvain(cgraph, resolution=resolution, max_iter=max_iter)
        cdf = self._sort_vertex_values(cdf, min_size)
        print(f"cugraph.louvain computed in {time.time() - subtic} seconds \n", 
              flush=True)
        return cdf, Q

    def compute_leiden(self, cgraph, resolution, max_iter=500, min_size=10):
        """ 9/Nov: Apparently fixed on branch 23.12: https://github.com/rapidsai/cugraph/issues/3749"""
        subtic = time.time()
        print(f"Performing Leiden on GPU: \n" \
              f"\t Resolution = {resolution}\n"
              f"\t Max iterations = {max_iter}\n"
              f"\t Min cluster size = {min_size}\n",
              flush=True)
        cdf, Q = self.cugraph.leiden(cgraph, resolution=resolution, max_iter=max_iter)
        cdf = self._sort_vertex_values(cdf, min_size)
        print(f"cugraph.leiden computed in {time.time() - subtic} seconds \n", 
              flush=True)
        print(f"Q = {Q}", flush=True)
        return cdf, Q
    
    def _sort_vertex_values(self, cdf, min_size):
        cdf = cdf.sort_values(by='vertex').partition.values
        cdf = self._sort_by_size(cdf, min_size)
        return cdf

    def _sort_by_size(self, clusters, min_size):
        """
        Relabel clustering in order of descending cluster size.
        New labels are consecutive integers beginning at 0
        Clusters that are smaller than min_size are assigned to -1.
        Copied from https://github.com/jacoblevine/PhenoGraph.

        Parameters
        ----------
        clusters: array
            Either numpy or cupy array of cluster labels.
        min_size: int
            Minimum cluster size.
        Returns
        -------
        relabeled: cupy array
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



class HybridPhenographModular():
    VALID_KNN_BACKENDS = ["CPU", "GPU"]
    VALID_REF_BACKENDS = ["CPU", "GPU"]
    VALID_GCLUST_BACKENDS = ["CPU", "GPU"] # Must be in cugraph
    VALID_GCLUSTERERS = ["louvain", "leiden"]
    """ Choose blocks and backends in a modular fashion. """
    def __init__(
            self, 
            knn="CPU", 
            refiner="CPU", 
            clusterer="CPU", 
            clustering="leiden"):
        self.knn = self.create_knn(knn)
        self.knn_backend = knn
        self.refiner = self.create_refiner(refiner)
        self.refiner_backend = refiner
        self.clusterer = self.create_clusterer(clusterer)
        self.clusterer_backend = clusterer
        self.clustering = clustering
        
    def create_knn(self, knn):
        if knn == "GPU":
            return KNNGPU()
        elif knn == "CPU":
            return KNNCPU()
        else:
            raise TypeError("Unsupported KNN type")
    
    def create_refiner(self, refiner):
        if refiner == "CPU":
            return JaccardRefinerCPU()
        elif refiner == "GPU":
            return JaccardRefinerGPU()
        else:
            raise TypeError("Unsupported refiner type")
    
    def create_clusterer(self, clusterer):
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
            two_pass_precision): 
        if self.knn_backend == "CPU":
            return self.knn.compute_neighbors(
                data, n_neighbors, algorithm, metric, p, n_jobs) # d, idx
        
        elif self.knn_backend == "GPU":
            return self.knn.compute_neighbors(
                data, n_neighbors, algorithm, metric, p, output_type=output_type, two_pass_precision=two_pass_precision) # Movement to cpu handled in refiner func checks.
        
        else:
            raise TypeError("Unsupported KNN type")
    
    def refiner_func(self, idx):
        if self.refiner_backend == "CPU" and self.knn_backend == "GPU": # IF GPU -> CPU
            return self.refiner.compute_jaccard(idx.get())
        else: # Within GPU 
            return self.refiner.compute_jaccard(idx) # jaccard edgelist
    
    def cluster_func(self, edgelist, resolution, min_size):
        if self.clusterer_backend == "CPU":
            graph = self.clusterer.edgelist_to_igraph(edgelist=edgelist) # leidenalg

        elif self.clusterer_backend == "GPU":
            graph = self.clusterer.edgelist_to_cugraph(edgelist=edgelist) # cugraph
            
        else:
            raise TypeError("Unsupported clusterer")
        
        # TODO: Run isolated node checks; latest branch includes isolated nodes in graph structure -> 24.04.6
        
        if self.clustering == "louvain":
            return self.clusterer.compute_louvain(graph, resolution=resolution, min_size=min_size)
        elif self.clustering == "leiden":
            return self.clusterer.compute_leiden(graph, resolution=resolution, min_size=min_size)
        else:
            raise TypeError("Unsupported clustering type")
    
    def cluster(
            self, 
            adata,
            n_neighbors,
            algorithm="auto",
            metric="euclidean",
            p=2,
            n_jobs=-1,
            two_pass_precision=False,
            resolution=1.0,
            min_size=10):
        """ Suitable for single cluster runs."""
        # Read in X_pca data
        embedding_name = "X_pca_harmony"
        if embedding_name not in adata.obsm:
            if "X_pca" not in adata.obsm:
                raise ValueError("No PCA embedding found in AnnData.obsm")
            embedding_name = "X_pca"
        data = adata.obsm[embedding_name]

        # KNN
        d, idx = self.knn_func(
            data, n_neighbors, algorithm, metric, p, "cupy", n_jobs, two_pass_precision) #

        # Jaccard
        refined_edgelist = self.refiner_func(idx)
        # Graph Clustering
        cdf, Q = self.cluster_func(
            refined_edgelist, resolution=resolution, min_size=min_size)
        return cdf, Q

class HybridPhenographSearch(HybridPhenographModular):
    def __init__(
            self, tmp_dir="./tmp", 
            knn="GPU", refiner="CPU", clusterer="GPU", clustering="leiden"):
        super().__init__(knn, refiner, clusterer, clustering)
        self.tmp_dir = tmp_dir
        self._make_temp_dir()
        self.param_grid = {}
        self.data_grid = {}
        self.quality_grid = {}
        if self.clusterer_backend == "GPU":
            self._set_array_backend("GPU")
            self._set_df_backend("GPU")
        else:
            self._set_array_backend("CPU")
            self._set_df_backend("CPU")
    
    def _set_array_backend(self, backend):
        """ Sets the dataframe backend for processing the clustering outputs. 
            If clusterer is GPU, will produce cudf outputs, so backend is set to
            cudf. Otherwise, pandas is used. """
        if backend == "GPU":
            try:
                import cupy 
                self.xp = cupy
            except ImportError:
                raise ImportError("cupy not installed.")
            
        elif backend == "CPU":
            try:
                import numpy 
                self.xp = numpy
            except ImportError:
                raise ImportError("numpy not installed.")
            
        else:
            raise TypeError("Unsupported backend")

    def _set_df_backend(self, backend):
        """ Sets the dataframe backend for processing the clustering outputs. 
            If clusterer is GPU, will produce cudf outputs, so backend is set to
            cudf. Otherwise, pandas is used. """
        if backend == "GPU":
            try:
                import cudf as cudf
                self.df = cudf
            except ImportError:
                raise ImportError("cudf not installed.")
            
        elif backend == "CPU":
            try:
                import pandas as pd
                self.df = pd
            except ImportError:
                raise ImportError("pandas not installed.")
            
        else:
            raise TypeError("Unsupported clusterer")
        
    def _make_temp_dir(self):
        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

    def _log_current_time(self):
        if self.log_time:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]", flush=True)
    
    def parameter_search(
            self, 
            adata,
            ks, 
            embedding_name="X_pca_harmony",
            algorithm="brute",
            metric="euclidean",
            p=2,
            n_jobs=-1,
            output_type="cupy",
            two_pass_precision=False,
            rs=[1], 
            min_size=10, 
            save=False,
            save_name="data",
            enable_cold_start=True, # NOT IMP
            cold_from=None,
            log_time=True): # NOT IMP
        """ Extension of self.cluster, over many Ks and Rs. No memory ops. All
        are cached, and should be loaded. """
        self.log_time = log_time
        print(f"Beginning Parameter Search... \n", flush=True)
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
            two_pass_precision=two_pass_precision)
        
        self._log_current_time()
        knn_name = f"indices_Kmax{max_k}_alg{algorithm}_metric{metric}_p{p}.npy"

        if save:
            # uncompressed for fast read write
            self.xp.save(f"{self.tmp_dir}/{knn_name}", idx)
        
        for k in ks:
            idx_subset = idx[:, :k]
            refined_edgelist = self.refiner_func(idx_subset)
            self._log_current_time()

            if save:
                jaccard_name = f"edgelist_jaccard_K{k}.feather"
                refined_edgelist.to_feather(
                    f"{self.tmp_dir}/{jaccard_name}")

            # Then for that Jaccard Graph, cluster for each resolution R
            for r in rs:
                clusters, Q = self.cluster_func(
                    refined_edgelist, resolution=r, min_size=min_size)
                self._log_current_time()

                # cache data
                clusters = clusters.tolist() # to cpu memory as py lists
                
                self.data_grid[(k, r)] = clusters # Back to memory if gpu rgardles
                self.quality_grid[(k, r)] = Q

                if save:
                    clusters_save = clusters.copy()
                    clusters_save.append(Q)
                    clustering_name = f"clusters_K{k}_r{r}.npy" # TODO: Ideally savez as two objects; Clusters and Q, rather than having Q in clusters array;
                    self.xp.save(f"{self.tmp_dir}/{clustering_name}", clusters_save)
                    del clusters_save
                    gc.collect()

            if save:
                del idx_subset
                del refined_edgelist
                gc.collect()

     # Create an index mapping of range indices to the actual indices
        number_index = [str(x) for x in range(adata.shape[0])]
        self.index_map = dict(zip(number_index, adata.obs.index))
        results = self.df.DataFrame(self.data_grid)
        results.index = results.index.astype(str)
        results.index = results.index.map(self.index_map) # index map -> cudf diff..
        
        adata = self._label_adata(adata, results)
        return adata
    
    def _label_adata(self, adata, results_df):
        """ Inplace operations on adata. All CPU based"""
        self._set_df_backend("CPU") # pandas

        # Store labels in an obsm matrix
        OBSM_ADDED_KEY = self.__class__.__name__ + "_labels"
         
        # Store what parameters each column label represents in the obsm matrix 
        UNS_ADDED_KEY_LABELMAP = self.__class__.__name__ + "_label_map" 
        label_map = {k:v for k,v in enumerate(self.data_grid.keys())} # obsm index : (k,r)

        adata.obsm[OBSM_ADDED_KEY] = results_df.values
        adata.uns[UNS_ADDED_KEY_LABELMAP] = label_map

        # Add param grids as quality grid
        UNS_ADDED_KEY_QUALITYSCORE = self.__class__.__name__ + "_quality_scores"
        df = self.df.DataFrame(
            self.df.DataFrame(
                self.quality_grid, 
                index=list(range(len(self.quality_grid)))).T[0])
        df = df.rename(columns={0:"modularity_score"})

        adata.uns[UNS_ADDED_KEY_QUALITYSCORE] = df

        return adata
    
    
# mimic searcher class for scanpy workflow

from datetime import datetime
from scanpy._utils import get_igraph_from_adjacency
import leidenalg as la

class ScanpyClustering():
    MAX_GPU_LEIDEN_ITER = 500
    VALID_BACKENDS = ["CPU", "GPU"]
    def __init__(self, backend="CPU"):
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
            except ImportError:
                raise ImportError("GPU backend requires rapids_singlecell package.")
        elif backend == "CPU":
            import scanpy as sc
            self.sc = sc
        else:
            raise ValueError("Backend must be either 'CPU' or 'GPU'.")
        
    
    def _set_array_backend(self, backend):
        """ Sets the dataframe backend for processing the clustering outputs. 
            If clusterer is GPU, will produce cudf outputs, so backend is set to
            cudf. Otherwise, pandas is used. """
        if backend == "GPU":
            try:
                import cupy 
                self.xp = cupy
            except ImportError:
                raise ImportError("cudf not installed.")
            
        elif backend == "CPU":
            try:
                import numpy 
                self.xp = numpy
            except ImportError:
                raise ImportError("pandas not installed.")
            
        else:
            raise TypeError("Unsupported clusterer")

    def _set_df_backend(self, backend):
        """ Sets the dataframe backend for processing the clustering outputs. 
            If clusterer is GPU, will produce cudf outputs, so backend is set to
            cudf. Otherwise, pandas is used. """
        if backend == "GPU":
            try:
                import cudf as cudf
                self.df = cudf
            except ImportError:
                raise ImportError("cudf not installed.")
            
        elif backend == "CPU":
            try:
                import pandas as pd
                self.df = pd
            except ImportError:
                raise ImportError("pandas not installed.")
            
        else:
            raise TypeError("Unsupported clusterer")
        
    
    def graph_cluster(self, adata, resolution, random_state=0, n_iter=-1):
        if self.backend == "CPU":
            # from scanpy src
            from scanpy._utils import get_igraph_from_adjacency as create_graph
            import leidenalg as la

            ig = create_graph(adata.obsp["connectivities"])
            w = np.array(ig.es["weight"]).astype(np.float64)
            part = la.find_partition(
                ig, 
                la.RBConfigurationVertexPartition, 
                weights=w, 
                seed=random_state, 
                resolution_parameter=resolution, 
                n_iterations=n_iter)
            groups = np.array(part.membership)
            Q = part.modularity
            return groups, Q
        
        elif self.backend == "GPU":
            # from rapids_singlecell src
            from rapids_singlecell.tools import _create_graph as create_graph
            from cugraph import leiden as culeiden 
            
            cg = create_graph(adata.obsp["connectivities"])
            if n_iter == -1:
                n_iter = self.MAX_GPU_LEIDEN_ITER # enforce max iters for gpu

            leiden_parts, Q = culeiden(
                cg,
                resolution=resolution,
                random_state=random_state,
                max_iter=n_iter,
            )

            # Format output
            groups = (
                leiden_parts.to_pandas().
                    sort_values("vertex")[["partition"]].to_numpy().ravel()
            )
            return groups, Q

        else:
            raise ValueError("Backend must be either 'CPU' or 'GPU'.")
        
    def evaluate_graph(self, graph):
        if self.backend == "CPU":
            nx_graph = graph.to_networkx()
        

        pass

class ScanpyClusteringSearch(ScanpyClustering):
    def __init__(self, backend="CPU"):
        super().__init__(backend)
        self.param_grid = {}
        self.data_grid = {}
        self.quality_grid = {}
    
    def _log_current_time(self, message):
        if self.log_time:
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n{message}", flush=True)
    
    def parameter_search(
            self,
            adata,
            ks,
            embedding_name="X_pca_harmony",
            n_pcs=None,
            rs=[1],
            random_state=0,
            n_iter=-1,
            min_size=10,
            log_time=True):
        """ Perform parameter search for scanpy clustering workflow. """
        self.log_time = log_time
        self._log_current_time("Starting parameter search")

        # Use PCA embeddings as input data 
        if embedding_name not in adata.obsm:
            if "X_pca" not in adata.obsm:
                self.sc.pp.pca(adata, n_comps=n_pcs)
            embedding_name = "X_pca"
        #data = adata.obsm[embedding_name]

        # Set up param grid
        self.param_grid["ks"] = ks
        self.param_grid["rs"] = rs
        
        adata.uns["param_grid"] = self.param_grid

        # Knn / neighbors
        for k in ks:
            self.sc.pp.neighbors(
                adata, 
                n_neighbors=10, 
                use_rep=embedding_name) #X_pca or X_pca_harmony
            self._log_current_time(f"Finished KNN with k={k}")
            
            for r in rs:
                labels, Q = self.graph_cluster(
                    adata, 
                    resolution=r, 
                    random_state=random_state,
                    n_iter=n_iter)
                
                #TODO -> min_size filtering
                #self.sc.tl.leiden(adata, resolution=r)
                self._log_current_time(f"\tFinished Leiden with resolution={r}")
                
                # Cache data
                clusters = labels.tolist()
                self.data_grid[(k, r)] = clusters
                self.quality_grid[(k, r)] = Q
        
        # Create an index mapping of range indices to the actual indices
        number_index = [str(x) for x in range(adata.shape[0])]
        self.index_map = dict(zip(number_index, adata.obs.index))
        results = self.df.DataFrame(self.data_grid)
        results.index = results.index.astype(str)
        results.index = results.index.map(self.index_map) # has no attr map?
        
        adata = self._label_adata(adata, results)
        return adata
    
    def _label_adata(self, adata, results_df):
        """ Inplace operations on adata. All CPU based"""
        self._set_df_backend("CPU") # pandas

        # Store labels in an obsm matrix
        OBSM_ADDED_KEY = self.__class__.__name__ + "_labels"
         
        # Store what parameters each column label represents in the obsm matrix 
        UNS_ADDED_KEY_LABELMAP = self.__class__.__name__ + "_label_map" 
        label_map = {k:v for k,v in enumerate(self.data_grid.keys())} # obsm index : (k,r)

        adata.obsm[OBSM_ADDED_KEY] = results_df.values
        adata.uns[UNS_ADDED_KEY_LABELMAP] = label_map

        # Add param grids as quality grid
        UNS_ADDED_KEY_QUALITYSCORE = self.__class__.__name__ + "_quality_scores"
        df = self.df.DataFrame(
            self.df.DataFrame(
                self.quality_grid, 
                index=list(range(len(self.quality_grid)))).T[0])
        df = df.rename(columns={0:"modularity_score"})

        adata.uns[UNS_ADDED_KEY_QUALITYSCORE] = df

        return adata
    
# from here anndataeval
from anndata import AnnData
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

class ClusteringSearchEvaluator():
    IMPLEMENTED_SEARCHERS = [
        "ScanpyClusteringSearch",
        "HybridPhenographSearch"
    ]
    """ Assess 'quality' of assigned cluster labels. """
    def __init__(self, adata: AnnData, searcher_name, gpu=False):
        """ Loads an AnnData
            cluster_labels should be a pd.DataFrame slice from .obs, where 
            columns contain the parameters, N rows down with a cluster label for 
            each element. """
        self.adata = adata

        assert searcher_name in self.IMPLEMENTED_SEARCHERS
        self.searcher_name = searcher_name
        self.cluster_labels = adata.obsm[searcher_name + "_labels"]
        self.cluster_label_map = adata.uns[searcher_name + "_label_map"]

        if gpu:
            self._set_ml_backend("cuml")
        else:
            self._set_ml_backend("sklearn")

    def _set_ml_backend(self, backend):
        """ Set the backend for the metrics. 
            Default is sklearn. 
            Options: sklearn, cuml"""
        
        if backend == "sklearn":
            import sklearn.metrics as metrics
        elif backend == "cuml":
            import cuml.metrics.cluster as metrics
        else:
            raise ValueError("Invalid backend. Options: sklearn, cuml")
        
        self.ml_backend = backend
        self.ml = metrics

    def get_K(self, k):
        """ For a given graph constructed from a KNN graph with K = k, 
            gets a N x R array, where each row is a cell, each column is a 
            resolution run, and values are cluster labels.
        """
        keys_with_k = {
            key:value[1] for key, value in self.cluster_label_map.items() 
                if value[0] == k}
        
        cluster_df = pd.DataFrame(
            self.cluster_labels[:, list(keys_with_k.keys())])
        cluster_df.columns = keys_with_k.values()
        cluster_df.columns.name = "R"

        return cluster_df
    
    def get_K_R(self, k, r):
        """ For a given clustering resolution r, knn value k, return the 
            cluster labels array"""
        return self.get_K(k)[r]
    
    def get_annotated_cluster_labels(self):
        # Get the nicer dataframe version of obsm
        cluster_df = pd.DataFrame(self.cluster_labels)
        cluster_df.columns = self.cluster_label_map.values()
        cluster_df.columns = cluster_df.columns.set_names(("K", "R"))
        return cluster_df
    
    """ Below fall into metrics which technically need a 'ground' truth. 
        (But here we specify the ground truth as the cluster labels themselves)"""
    def between_model_score(self, score_function, k_only=False, r_only=False, **kwargs):
        """ For a N x P matrix, 
            Compare the score of a clustering run P, against other Ps.
            Return a P x P array."""
        df = self.get_annotated_cluster_labels()
        p_len = df.shape[1]
        pp_matrix = pd.DataFrame(
            np.zeros((p_len, p_len)), 
            index = df.columns, 
            columns = df.columns)
        
        for i in df.columns:
            for j in df.columns:
                pp_matrix.loc[i, j] = score_function(
                    df.loc[:, i].values, 
                    df.loc[:, j].values,
                    **kwargs)
        return pp_matrix

    def adjusted_rand_index(self):
        """ For a N x K matrix, 
            Return a P x P array of adjusted rand indices for each P. """
        return self.between_model_score(
            self.ml.adjusted_rand_score)
    
    def normalized_mutual_info(self):
        """ For a N x K matrix, 
            Return a P x P array of normalized mutual info scores for each P. """
        return self.between_model_score(
            self.ml.normalized_mutual_info_score)
    
    def adjusted_mutual_info(self):
        """ For a N x K matrix, 
            Return a P x P array of adjusted mutual info scores for each parameter run P. """
        return self.between_model_score(
            self.ml.adjusted_mutual_info_score)
    
    def rank_parameters(self, pp_matrix):
        pass

    def stability_score(self, df, penalty_weight=0.001):
        stability_scores = {}
        for param in df.columns:
            k, r = param
            other_params = [p for p in df.columns if p != param]
            mean_ami = df.loc[param, other_params].mean()
            penalty = penalty_weight * k  # Penalize higher K values
            stability_scores[param] = mean_ami - penalty
        ranked_stability_scores = sorted(
            stability_scores.items(), key=lambda x: x[1], reverse=True)
        return ranked_stability_scores

    def dotplot(self):
        import scanpy as sc
        adata_copy = self.adata.copy()
        adata_copy.obs["_tmp_lbl"] = list(self.get_K_R(10, 0.5))
        adata_copy.obs["_tmp_lbl"] = adata_copy.obs["_tmp_lbl"].astype("category")
        sc.pl.dotplot(adata_copy, adata_copy.var_names, "_tmp_lbl")

    def matrixplot(self):
        import scanpy as sc
        adata_copy = self.adata.copy()
        adata_copy.obs["_tmp_lbl"] = list(self.get_K_R(10, 0.5))
        adata_copy.obs["_tmp_lbl"] = adata_copy.obs["_tmp_lbl"].astype("category")
        sc.pl.dotplot(adata_copy, adata_copy.var_names, "_tmp_lbl")

""" Master classes """

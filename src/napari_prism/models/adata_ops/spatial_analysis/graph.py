from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

try:
    import cugraph
except ImportError:
    cugraph = None

from typing import Literal

from anndata import AnnData


@dataclass
class GraphWrapper:
    csr: sp.csr_matrix | None = field(default=None)
    dense: np.ndarray | None = field(default=None)
    nx_graph: nx.Graph | None = field(default=None)
    cugraph_graph: "cugraph.Graph" | None = field(default=None)  # type: ignore

    # convert lazily
    def to_csr(self):
        if self.csr is None:
            if self.dense is not None:
                self.csr = sp.csr_matrix(self.dense)
            elif self.nx_graph is not None:
                self.csr = nx.to_scipy_sparse_matrix(
                    self.nx_graph, format="csr"
                )
            elif self.cugraph_graph is not None:
                # Convert cuGraph to CSR (device to host)
                edges_df = self.cugraph_graph.view_edge_list()
                n = max(edges_df["src"].max(), edges_df["dst"].max()) + 1
                mat = sp.csr_matrix(
                    (
                        edges_df["weight"].to_numpy(),
                        (
                            edges_df["src"].to_numpy(),
                            edges_df["dst"].to_numpy(),
                        ),
                    ),
                    shape=(n, n),
                )
                self.csr = mat
        return self.csr

    def to_dense(self):
        if self.dense is None:
            if self.csr is not None:
                self.dense = self.csr.toarray()
            elif self.nx_graph is not None:
                self.dense = nx.to_numpy_array(self.nx_graph)
            elif self.cugraph_graph is not None:
                self.dense = self.to_csr().toarray()
        return self.dense

    def to_networkx(self):
        if self.nx_graph is None:
            if self.csr is not None:
                self.nx_graph = nx.from_scipy_sparse_matrix(self.csr)
            elif self.dense is not None:
                self.nx_graph = nx.from_numpy_array(self.dense)
            elif self.cugraph_graph is not None:
                self.nx_graph = nx.from_scipy_sparse_matrix(self.to_csr())
        return self.nx_graph

    def to_cugraph(self):
        if cugraph is None:
            raise ImportError("cugraph not installed")
        if self.cugraph_graph is None:
            if self.csr is not None:
                import cudf

                coo = self.csr.tocoo()
                edges_df = cudf.DataFrame(
                    {"src": coo.row, "dst": coo.col, "weight": coo.data}
                )
                self.cugraph_graph = cugraph.Graph()
                self.cugraph_graph.from_cudf_edgelist(
                    edges_df,
                    source="src",
                    destination="dst",
                    edge_attr="weight",
                )
            elif self.nx_graph is not None:
                return self.from_networkx_to_cugraph(self.nx_graph)
            elif self.dense is not None:
                return self.from_dense_to_cugraph(self.dense)
        return self.cugraph_graph


def symmetrise_graph(adjacency_matrix):
    """Symmetrise a graph adjacency matrix"""
    # Symmetrize graph (make undirected); if A->B, then enforce B->A
    # Rather than dividing by 2, clip and sign to enforce 0/1
    if isinstance(adjacency_matrix, np.ndarray):
        sym = adjacency_matrix + adjacency_matrix.T
        sym = np.clip(sym, 0, 1)

    elif isinstance(adjacency_matrix, scipy.sparse.csr.csr_matrix):
        sym = adjacency_matrix + adjacency_matrix.T
        sym = sym.sign()

    else:
        raise ValueError("invalid adjacency matrix type")

    return sym


# Concrete Functions; gcross may be a class
def get_closest_neighbors(
    spatial_coordinates: np.ndarray,
    query_indices: np.ndarray | list,
    target_indices: np.ndarray | list,
):
    """
    Given a matrix of spatial coordinates in euclidean space, i.e. N x 2, get
    the closest neighbors for each query index in the target indices.

    Returns:
        distances: N x 1 array of distances to the closest neighbor, where
            N is the number of query indices.
        indices: N x 1 array of indices of the closest neighbor in the target indices.

    """
    if set(query_indices) == set(target_indices):
        # If query and target indices are the same, use nearest neighbors
        # to get the closest neighbor excluding self.
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(spatial_coordinates[target_indices])
        distances, indices = nn.kneighbors(spatial_coordinates[query_indices])
        distances = distances[:, 1]  # exclude self distts
        indices = indices[:, 1]  # exclude self indices

    else:
        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(spatial_coordinates[target_indices])
        distances, indices = nn.kneighbors(spatial_coordinates[query_indices])

    return distances, indices


def compute_pair_interactions(
    adata: AnnData,
    phenotype_column: str,
    phenotype_A: str,
    phenotype_B: str,
    method: Literal["nodes", "edges"],
    adjacency_matrix: np.ndarray | scipy.sparse.csr.csr_matrix = None,
    connectivity_key: str = "spatial_connectivities",
) -> tuple[int, int, bool]:
    """
    Uses adjacency_matrix first if supplied, otherwise tries to find
    adjacency matrix in adata.obsp using `connectivity_key`.

    Compute the number of interactions between two phenotypes in a graph.
    Enforced symmetric relations. i.e.) IF A -> B, then B -> A.

    If neighbors graph constructed with radius, then already symmetric.

    Returns:
    total_interactions: Number of interactions between phenotype_pair
    total_cells: Total number of cells in the graph
    missing: True if not enough cells for comparison
    """
    adata = adata.copy()
    adata.obs = adata.obs.reset_index()
    if adjacency_matrix is None:
        if connectivity_key not in adata.obsp:
            raise ValueError(
                "No adjacency matrix provided and no "
                "connectivity key found in adata.obsp"
            )
        else:
            adjacency_matrix = adata.obsp[connectivity_key]

    sym = symmetrise_graph(adjacency_matrix)
    a_ix = list(
        adata.obs[adata.obs[phenotype_column] == phenotype_A].index.astype(int)
    )
    b_ix = list(
        adata.obs[adata.obs[phenotype_column] == phenotype_B].index.astype(int)
    )
    ab = sym[np.ix_(a_ix, b_ix)]  # A rows -> B cols
    ba = sym[np.ix_(b_ix, a_ix)]  # B rows -> A cols

    total_cells = sum(ab.shape)

    # Count the number of nodes of pair A and B that neighbor each other / totals
    if method == "nodes":
        if isinstance(adjacency_matrix, np.ndarray):
            f_sum = ab.any(
                axis=1
            ).sum()  # How many A cells have atleast 1 B neighbor
            s_sum = ba.any(
                axis=1
            ).sum()  # How many B cells have atleast 1 A neighbor

        elif isinstance(adjacency_matrix, csr_matrix):
            f_sum = (ab.getnnz(axis=1) > 0).sum()
            s_sum = (ba.getnnz(axis=1) > 0).sum()

        else:
            raise ValueError("invalid adjacency matrix type")

        total_interactions = (
            f_sum + s_sum
        )  # Represents total number of interacting cells in A and B

    # Count the number of times pair A and B neighbor each other / totals
    elif method == "edges":
        f_sum = (
            ab.sum()
        )  # How many B neighbors every A cells have in the graph
        s_sum = (
            ba.sum()
        )  # How many A neighbors every B cells have in the graph

        total_interactions = (
            f_sum + s_sum
        )  # Represents total number of interactions between A and B

    else:
        raise ValueError("invalid method")

    # Account for self comparisons. Normalised by density, but need to report counts
    if phenotype_A == phenotype_B:
        total_interactions = total_interactions / 2
        total_cells = total_cells / 2

    # # Minimum number of cells for a comparison
    not_enough_cells = total_cells < 2
    # For different phenotypes. If self, then not_enough_cells will be 0 anyway
    not_enough_of_category = len(a_ix) == 0 or len(b_ix) == 0

    missing = False
    if not_enough_cells or not_enough_of_category:
        missing = True

    return total_interactions, total_cells, missing


# Pairwise Cell Computations
def compute_targeted_degree_ratio(
    adata,
    adjacency_matrix,
    phenotype_column,
    phenotype_A,  # source phenotype
    phenotype_B,  # target phenotype
    directed=False,
):
    """For each node in the adjacency matrix, compute the ratio of its
    targets that are of phenotype_pair.

    If directed, then this becomes the outdegree ratio. i.e.) If
    KNN, then the score is the ratio of its closest K neighbors being of
    the other specified type.

    If not directed, then this becomes a simple degree ratio, with the graph
    being symmetrised (enforce A->B, then B->A).

    """
    mat = adjacency_matrix if directed else symmetrise_graph(adjacency_matrix)

    a_mask = adata.obs[phenotype_column] == phenotype_A
    b_mask = adata.obs[phenotype_column] == phenotype_B
    a_ix = list(np.where(a_mask)[0])
    b_ix = list(np.where(b_mask)[0])
    a = mat[a_ix]  # A rows -> all cols
    ab = mat[np.ix_(a_ix, b_ix)]  # A rows -> B cols

    a_edge_degrees = a.sum(axis=1)  # Total connections for each A cell
    a_target_degrees = ab.sum(
        axis=1
    )  # Total connections to B cells for each A cell

    a_ab = np.divide(
        a_target_degrees, a_edge_degrees
    )  # For each A cell, ratio of B connections to total connections

    return a_ab

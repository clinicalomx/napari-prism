# Cells
from typing import List
import anndata as ad
import numpy as np
import scipy
from ._utils import symmetrise_graph, timer_func, normalise_log2p, annotate_tree

# Nhoods
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import time
from sklearn.cluster import MiniBatchKMeans, KMeans
import seaborn as sns
from anndata import AnnData
from scipy.stats import ttest_ind
import statsmodels.api as sm
from .._anndata_helpers import ObsHelper
from ._constants import NOLAN_KEY, CN_KEY, NE_KEY

def compute_targeted_degree_ratio(
        adata, 
        adjacency_matrix,
        phenotype_column, 
        phenotype_A, # source phenotype
        phenotype_B, # target phenotype
        directed=False):
    """ For each node in the adjacency matrix, compute the ratio of its
        targets that are of phenotype_pair. 

        If directed, then this becomes the outdegree ratio. i.e.) If
        KNN, then the score is the ratio of its closest K neighbors being of 
        the other specified type.

        If not directed, then this becomes a simple degree ratio, with the graph
        being symmetrised (enforce A->B, then B->A).
        
    """
    if directed:
        mat = adjacency_matrix
    else:
        mat = symmetrise_graph(adjacency_matrix)

    a_ix = list(adata.obs.query(f"{phenotype_column} == '{phenotype_A}").index.astype(int))
    b_ix = list(adata.obs.query(f"{phenotype_column} == '{phenotype_B}").index.astype(int))
    a = mat[a_ix] # A rows -> all cols
    ab = mat[np.ix_(a_ix, b_ix)] # A rows -> B cols

    a_edge_degrees = a.sum(axis=1) # Total connections for each A cell
    a_target_degrees = ab.sum(axis=1) # Total connections to B cells for each A cell
    
    a_ab = np.divide(a_target_degrees, a_edge_degrees) # For each A cell, ratio of B connections to total connections

    return a_ab

""" Neighborhoods """
def _get_neighborhoods_from_job(
        job, 
        region_groupby, 
        knn, 
        x_coordinate: str,
        y_coordinate: str,
        z_coordinate: str):
        """ For a given job (i.e. a given region/image in the dataset), return the indices of the nearest neighbors, for each cell in that job.
        Called by process_jobs.
            
            Params:
                job (str): Metadata containing start time, index of reigon, region name, indices of reigon in original dataframe.
                n_neighbors (str): Number of neighbors to find for each cell. 
            
            Returns:
                neighbors (numpy.ndarray): Array of indices where each row corresponds to a cell, and values correspond to indices of the nearest neighbours found for that cell. Sorted.
        """
        # Unpack job metadata file
        _, region, indices = job
        region = region_groupby.get_group(region)

        if z_coordinate is None:
            coords = [x_coordinate, y_coordinate]
        else:
            coords = [x_coordinate, y_coordinate, z_coordinate]

        X_data = region.loc[indices][coords].values
        # Perform sklearn unsupervised nearest neighbors learning, on the x y coordinate values
        # Essentially does euclidean metric; but technically Minkowski, with p=2
        neighbors = NearestNeighbors(n_neighbors=knn).fit(X_data)
        # Unpack results
        distances, indices = neighbors.kneighbors(X_data)
        sorted_neighbors = _sort_neighbors(region, distances, indices)
        return sorted_neighbors.astype(np.int32)

def _sort_neighbors(region, distances, indices):
    """ Processes the two outputs of sklearn NearestNeighbors to sort indices of nearest neighbors. """
    # Sort neighbors
    args = distances.argsort(axis = 1)
    add = np.arange(indices.shape[0])*indices.shape[1]
    sorted_indices = indices.flatten()[args+add[:,None]]
    neighbors = region.index.values[sorted_indices]
    return neighbors

def cellular_neighborhoods_sq(
    adata,
    phenotype: str,
    connectivity_key: str,
#    library_key: str | None = None,
    k_kmeans: List[int] = [10],
    mini_batch_kmeans: bool = True,
    parallelise: bool = False
):
    phenotypes = adata.obs[phenotype].unique()

    conn = adata.obsp[connectivity_key]
    row_ix, col_ix = conn.nonzero()
    # List incase of ragged arr -> i.e. if graph is not symmetric.
    neighbors = [[] for _ in range(conn.shape[0])]

    # For each row or cell, get its neighbors according to the graph;
    cell_indices = adata.obs.index
    for r in range(conn.shape[0]):
        cix = np.where(row_ix == r)
        neighbors[r] = col_ix[cix]
    
    X_dat = adata.obs
    dummies = pd.get_dummies(X_dat[phenotype])
    dummy_cols = dummies.columns
    dummies_np = dummies.values

    counted_neighbors = np.zeros((conn.shape[0], dummies_np.shape[1]), dtype=int)
    for i, neighbor_indices in enumerate(neighbors):
        if neighbor_indices.size > 0:
            counted_neighbors[i] = dummies_np[neighbor_indices].sum(axis=0)

    total_neighbor_counts = pd.DataFrame(
        counted_neighbors,
        columns=dummy_cols,
        index=cell_indices
    )

    # Reannotate the frequency graph; technically these can be in obsm
    total_neighbor_counts.columns.name = phenotype
    adata.obsm["neighbor_counts"] = total_neighbor_counts
    print("Counts done")

    # Below represnet distinct following step in workflow; KMeans
    kmeans_cls = MiniBatchKMeans if mini_batch_kmeans else KMeans
    
    kmeans_instance = None
    labels = []
    inertias = []
    enrichment_scores = {}
    print("Starting KMeans loop")
    for k in k_kmeans:
        print(k)
        # Instantiate kmeans instance
        if kmeans_instance is not None:
            kmeans_instance.n_clusters = k
        else:
            kmeans_instance = kmeans_cls(
                n_clusters=k, 
                n_init=10, 
                random_state=0,
                init="k-means++" # 'best' initializer for kms
                )
        
        # first
        y = kmeans_instance.fit_predict(total_neighbor_counts.values)

        # enrichment scores;
        distances_to_centroids = kmeans_instance.cluster_centers_
        frequencies = total_neighbor_counts.mean(axis=0).values
        num = distances_to_centroids + frequencies
        norm = (distances_to_centroids + frequencies).sum(axis=1, keepdims=True)
        score = np.log2(num/norm/frequencies)
        score_df = pd.DataFrame(
            score, 
            columns=pd.Index(phenotypes, name=phenotype))
        score_df.index.name = "CN_index"

        enrichment_scores[str(k)] = (score_df)
        inertias.append(kmeans_instance.inertia_)
        labels.append(y)
    
    # Store in DataArray-like format
    # matrices are ragged so data is a dictionary.
    adata.uns["cn_enrichment_matrices"] = enrichment_scores
    adata.uns["cn_enrichment_matrices_dims"] = {
        "k_kmeans": k_kmeans}

    cn_labels = np.array(labels).T
    # structured
    cn_labels = np.array(
        cn_labels, 
        dtype=[("k_kmeans", cn_labels.dtype)])

    adata.obsm["cn_labels"] = cn_labels
    adata.uns["cn_labels_dims"] = {
        "k_kmeans": k_kmeans
    }

    cn_inertias = pd.DataFrame(
        inertias,
        columns=["Inertia"],
        index=pd.Index(k_kmeans, name="k_kmeans")
    )
    adata.uns["cn_inertias"] = cn_inertias

@timer_func
def cellular_neighborhoods(
        adata,
        grouping: str,
        x_coordinate: str = "x",
        y_coordinate: str = "y",
        z_coordinate: str | None = None, # TODO: z-coordinate support?
        knn: int = 10,
        kmeans: int = 10,
        phenotype: str = None,
        inplace: bool = True
    ):
    # 1) Generate jobs
    # Extract coordinate, phenotype, and grouping data

    if z_coordinate is not None:
        data = adata.obs[[x_coordinate, y_coordinate, z_coordinate, phenotype, grouping]]
    else:
        data = adata.obs[[x_coordinate, y_coordinate, phenotype, grouping]]
    
    index_map = dict(zip(list(data.index), list(data.reset_index().index)))
    data = data.reset_index()
    
    # One-hot encode data
    data = pd.concat([data, pd.get_dummies(data[phenotype])], axis=1) # Dummifie the cluster column into one-hot encoding

    # Transpose/Flatten phenotype data
    phenotypes = data[phenotype].unique()
    data_summed_phenotype = data[phenotypes].values # Tranposed + One-hot encoded phenotypes

    # Groupby object, groupby regions
    region_groupby = data.groupby(grouping)

    # All the different regions
    regions = list(data[grouping].unique())

    # 1) Generate jobs. Indices are the dataframe indices for the given tissue/region
    jobs = [
        (regions.index(t),t,a) for t, indices in region_groupby.groups.items() \
        for a in np.array_split(indices,1)
        ]
    
    # 2) Process jobs
    processed_neighbors = list()
    for job in jobs:
        processed_neighbors.append(
            _get_neighborhoods_from_job(
                job, 
                region_groupby, 
                knn, 
                x_coordinate, 
                y_coordinate, 
                z_coordinate
                )
            )

    # 3) Annotate jobs
    out_dict = {}
    for neighbors, job in zip(processed_neighbors, jobs):
        chunk = np.arange(len(neighbors))
        region_name = job[1]
        indices = job[2]
        within = neighbors[chunk, :knn] #query up to knn neighbors
        window = data_summed_phenotype[within.flatten()]
        window = window.reshape(len(chunk), knn, len(phenotypes)).sum(axis = 1)
        out_dict[region_name] = (window.astype(np.float16), indices)

    region_dfs = [
        pd.DataFrame(
            out_dict[region_name][0],
            index = out_dict[region_name][1].astype(int),
            columns = phenotypes) 
        for region_name in regions  
        ]
    
    window = pd.concat(region_dfs, axis=0)
    window = window.loc[data.index.values]
    if z_coordinate is None:
        keep_cols = [x_coordinate, y_coordinate, phenotype]
    else:
        keep_cols = [x_coordinate, y_coordinate, z_coordinate, phenotype]
    window = pd.concat([data[keep_cols], window], axis=1) # Except grouping

    # Then perform knn on window
    kmeans_neighborhoods = MiniBatchKMeans(n_clusters=kmeans, n_init=3, random_state=0)
    X_data = window[phenotypes]#.values
    kmeans_neighborhoods_labels = kmeans_neighborhoods.fit_predict(X_data.values)
    
    # Compute enrichment scores;
    phenotype_distance_to_CN_centroids = kmeans_neighborhoods.cluster_centers_
    phenotype_frequencies = data_summed_phenotype.mean(axis=0) # Frequency of cell type across entire dataset. == Cells that are phenotype X / Total Cells
    num = phenotype_distance_to_CN_centroids + phenotype_frequencies # n neighbors rows, phenotype cols
    norm = (phenotype_distance_to_CN_centroids + phenotype_frequencies).sum(axis=1, keepdims=True) 
    score = np.log2(num/norm/phenotype_frequencies)
    score_df = pd.DataFrame(score, columns=phenotypes)

    results = {
        "labels": kmeans_neighborhoods_labels,
        "inertia": kmeans_neighborhoods.inertia_,
        "enrichment_matrix": score_df
    }
    # Annotate assignments back to AnnData using a tree-like structure
    # with a nested dicts format; indexed by the data it shows
    # Splits in nodes by different parameters
    if inplace:
        adata.obsm[f"nolan_{CN_KEY}_input_X"] = X_data.values # Save the X matrix used for kmeans

        if NOLAN_KEY not in adata.uns:
            adata.uns[NOLAN_KEY] = dict()

        if CN_KEY not in adata.uns[NOLAN_KEY]:
            adata.uns[NOLAN_KEY][CN_KEY] = dict()

        uns_node = adata.uns[NOLAN_KEY][CN_KEY]
        
        for k, v in results.items():
            node = annotate_tree(
                start_node=uns_node,
                node_label_A=k,
                node_label_B=(grouping, phenotype),
                data=dict())
            
            annotate_tree(
                start_node=node,
                node_label_A=(knn, kmeans),
                data=v)

    else:
        return results


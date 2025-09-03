"""https://doi.org/10.1016/j.cell.2020.10.021"""

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans, MiniBatchKMeans


def cellular_neighborhoods_sq(
    adata,
    phenotype: str,
    connectivity_key: str,
    #    library_key: str | None = None,
    k_kmeans: list[int] = None,
    mini_batch_kmeans: bool = True,
    parallelise: bool = False,
) -> None:
    """
    Compute Nolan's cellular neighborhoods compatible with squidpy-generated
    .obsp spatial graphs. By default, stores the results inplace.

    Args:
        adata: AnnData object.
        phenotype: Cell label to compute neighborhoods on.
        connectivity_key: Key for the adjacency matrix in adata.obsp. Ideally,
            should be a KNN graph to stay true to the original definition of
            cellular neighborhoods.
        k_kmeans: List of K values to use for KMeans clustering. If None,
            defaults to [10].
        mini_batch_kmeans: If True, uses MiniBatchKMeans instead of KMeans.
    """
    if k_kmeans is None:
        k_kmeans = [10]

    # phenotypes = adata.obs[phenotype].unique().dropna()

    conn = adata.obsp[connectivity_key]
    # row_ix, col_ix = conn.nonzero()
    # List incase of ragged arr -> i.e. if graph is not symmetric.
    neighbors = [[] for _ in range(conn.shape[0])]

    # For each row or cell, get its neighbors according to the graph;
    cell_indices = adata.obs.index
    # for r in range(conn.shape[0]):
    #     cix = np.where(row_ix == r)
    #     neighbors[r] = col_ix[cix]

    # speed up with csr row ptrs
    neighbors = [
        conn.indices[conn.indptr[i] : conn.indptr[i + 1]]
        for i in range(conn.shape[0])
    ]

    X_dat = adata.obs
    dummies = pd.get_dummies(X_dat[phenotype])
    dummy_cols = dummies.columns
    dummies_np = dummies.values

    counted_neighbors = np.zeros(
        (conn.shape[0], dummies_np.shape[1]), dtype=int
    )
    for i, neighbor_indices in enumerate(neighbors):
        if neighbor_indices.size > 0:
            counted_neighbors[i] = dummies_np[neighbor_indices].sum(axis=0)

    total_neighbor_counts = pd.DataFrame(
        counted_neighbors, columns=dummy_cols, index=cell_indices
    )

    # Reannotate the frequency graph; technically these can be in obsm
    total_neighbor_counts.columns.name = phenotype
    adata.obsm["neighbor_counts"] = total_neighbor_counts
    logger.info("Neighbor phenotype counts done")

    # Below represnet distinct following step in workflow; KMeans
    kmeans_cls = MiniBatchKMeans if mini_batch_kmeans else KMeans

    kmeans_instance = None
    labels = []
    inertias = []
    enrichment_scores = {}
    logger.info("Starting KMeans loop")
    for k in k_kmeans:
        logger.info(k)
        # Instantiate kmeans instance
        if kmeans_instance is not None:
            kmeans_instance.n_clusters = k
        else:
            kmeans_instance = kmeans_cls(
                n_clusters=k,
                n_init=3,
                random_state=0,
                init="k-means++",  # 'best' initializer for kms
            )

        # first
        y = kmeans_instance.fit_predict(total_neighbor_counts.values)

        # enrichment scores;
        distances_to_centroids = kmeans_instance.cluster_centers_
        # frequencies = total_neighbor_counts.astype(bool).mean(axis=0).values
        frequencies = (
            dummies[total_neighbor_counts.columns].mean(axis=0).values
        )
        num = distances_to_centroids + frequencies
        norm = (distances_to_centroids + frequencies).sum(
            axis=1, keepdims=True
        )
        score = np.log2(num / norm / frequencies)
        score_df = pd.DataFrame(
            score,
            columns=pd.Index(total_neighbor_counts.columns, name=phenotype),
        )
        score_df.index.name = "CN_index"

        enrichment_scores[str(k)] = score_df
        inertias.append(kmeans_instance.inertia_)
        labels.append(y)

    # Store in DataArray-like format
    # matrices are ragged so data is a dictionary.
    adata.uns["cn_enrichment_matrices"] = enrichment_scores
    adata.uns["cn_enrichment_matrices_dims"] = {"k_kmeans": k_kmeans}

    cn_labels = pd.DataFrame(np.array(labels).T)
    cn_labels.columns = k_kmeans
    cn_labels.columns = cn_labels.columns.astype(str)
    cn_labels.index = adata.obs.index
    # structured
    # cn_labels = np.array(cn_labels)#, dtype=[("k_kmeans", cn_labels.dtype)])

    adata.obsm["cn_labels"] = cn_labels
    adata.uns["cn_labels_dims"] = {"k_kmeans": k_kmeans}

    cn_inertias = pd.DataFrame(
        inertias,
        columns=["Inertia"],
        index=pd.Index(k_kmeans, name="k_kmeans"),
    )
    adata.uns["cn_inertias"] = cn_inertias

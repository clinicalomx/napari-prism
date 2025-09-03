"""https://doi.org/10.1101/2024.03.26.586902"""

# def hoodscan_sq(
#     adata: AnnData, phenotype: str, distances_key: str, tau: float
# ) -> None:
#     # Retrieve knn dists from sq;
#     dists = adata.obsp[distances_key]


# import squidpy as sq

# sq.gr.spatial_neighbors(
#     adata, n_neighs=100, coord_type="generic", library_key="patientid"
# )
# phenotype = "celltypes"
# conn = adata.obsp["spatial_distances"]
# neighbors = [[] for _ in range(conn.shape[0])]
# # For each row or cell, get its neighbors according to the graph;
# cell_indices = adata.obs.index
# # for r in range(conn.shape[0]):
# #     cix = np.where(row_ix == r)
# #     neighbors[r] = col_ix[cix]

# # speed up with csr row ptrs; this should be C by K
# neighbors = [
#     conn.indices[conn.indptr[i] : conn.indptr[i + 1]]
#     for i in range(conn.shape[0])
# ]

# # also log distances
# distances = [conn.data[ptrs] for ptrs in neighbors]
# distances = np.array(distances)


# # convert to probs using softmax;, governed by tau; median as a std
# tau = np.median(distances)

# X_dat = adata.obs
# dummies = pd.get_dummies(X_dat[phenotype])
# dummy_cols = dummies.columns
# dummies_np = dummies.values

# counted_neighbors = np.zeros((conn.shape[0], dummies_np.shape[1]), dtype=int)
# for i, neighbor_indices in enumerate(neighbors):
#     if neighbor_indices.size > 0:
#         counted_neighbors[i] = dummies_np[neighbor_indices].sum(axis=0)

# total_neighbor_counts = pd.DataFrame(
#     counted_neighbors, columns=dummy_cols, index=cell_indices
# )

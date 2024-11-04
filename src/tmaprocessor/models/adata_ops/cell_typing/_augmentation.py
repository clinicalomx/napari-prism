# # Banksy
# # Set backends 
import anndata as ad
import pandas as pd
import numpy as np 
# import cupyx.scipy.stats as cpxstats
# import cupy as cp
# import rapids_singlecell as rsc
# import cupyx.scipy.sparse as cpx
# import cuml

""" Data Augmentation classes; Usually expansion along .var dimension """

def add_obs_as_var(adata, obs_columns, layer=None):
    """ Expansion along .var dimension, by adding .obs as a feature. 
        Return new obj. """
    # Adds an observation col as a .var feature
    if layer is not None:
        X = adata.X
    else:
        X = adata.layers[layer]
    
    assert all([x in adata.obs.columns for x in obs_columns])

    if isinstance(obs_columns, str):
        obs_columns = [obs_columns]


    aug_X = np.hstack((X, adata.obs[obs_columns].values.reshape(-1, len(obs_columns))))
    var_names = adata.var_names
    obs_columns_varred = [f"{x}_var" for x in obs_columns]
    var_names = var_names.append(pd.Index(obs_columns_varred))
    # Inherit all attrs; except varm and layers which miss the added vars
    aug_adata = ad.AnnData(
        aug_X,
        obs=adata.obs,
        uns=adata.uns,
        obsm=adata.obsm,
    )
    aug_adata.var.index = var_names
    aug_adata.layers["augmented_X"] = aug_adata.X
    return aug_adata

def subset_adata_by_var(adata, var_subset):
    """ Contraction along .var dimension, by subsetting .var. 
        Return view. """
    # Subset adata by var names
    assert all([x in adata.var_names for x in var_subset])
    return adata[:, var_subset]

class BanksyCPU():
    pass

# class BanksyGPU():
#     def __init__(self):
#         self.check_if_GPU_version_installed()

#     def check_if_GPU_version_installed(self):
#         try:
#             import cuml
#             self.cuml = cuml
#         except ImportError:
#             raise ImportError("cuml not installed. \
#                               Need to install gpu extras of this package: \
#                               `pip install 'pcf_analysis_poetry[gpu]'`")
        
#         try:
#             import cupy
#             self.cupy = cupy
#         except ImportError:
#             raise ImportError("cupy not installed. \
#                               Need to install gpu extras of this package: \
#                               `pip install 'pcf_analysis_poetry[gpu]'`")
        
#     def csr_row_normalisation(cp_csr):
#         """ L1 norm using cuml; row normalisation such that values in csr matrix sum up to 1
#         """
#         # unit_vec = cp.ones(cp_csr.shape[1], dtype=cp.float32)
#         # d_u = cp_csr.dot(unit_vec).reshape(cp_csr.shape[1], 1)
#         # L1 norm; using cuml
#         from cuml.preprocessing import Normalizer
#         transformer = Normalizer(norm="l1").fit(cp_csr)
#         return transformer.transform(cp_csr)

#     def generate_azimuth_matrix(locs, dists):
#         # Given a list of coordinates, the distances csr; compute the azimuthal angles between knns
#         # From the spatial graph -> extract connected pairs of cells 
#         # Given these pairs, get their coordinates from locs, then compute the azimuthal angles between them
#         """ Should be equivalent to theta_from_spatial_graph. Given a list of coordinates in Real coordinate space (x,y), locs, 
#             and 2) the spatial graph or dists between cells, compute the azimuth / theta matrix as a CSR. 
#             In the paper they define coordinates as polar, anti-clockwise from horizontal. """
#         # Assuming your distance matrix is named 'dists' (scipy.csr).

#         # Cast to a cupyx
#         dists = cpx.csr_matrix(dists)
        
#         # Extract non-zero values and their corresponding row and column indices; use scipy funcs
#         nonzero_rows, nonzero_cols, nonzero_values = cpx.find(dists)
        
#         # Extract coordinates for the nonzero indices
#         nonzero_coordinates = cp.array([locs[nonzero_rows], locs[nonzero_cols]])
        
#         # Calculate differences in coordinates
#         deltas = nonzero_coordinates[1] - nonzero_coordinates[0]
        
#         # Compute angles using cp.arctan2
#         angles = cp.arctan2(deltas[:, 1], deltas[:, 0])
        
#         # Create a csr_matrix with the same shape as dists
#         thetas = cpx.csr_matrix((angles, (nonzero_rows, nonzero_cols)), shape=dists.shape, dtype=cp.float32)

#         return thetas

#     # Then if M >=1, they apply an azimuthal gabor filter 
#     def agf_transform(weights, m, theta):
#         """ Azimuthal Fourier Transform, augment with 'frequencies' 
#             of the azimuthal angles. """
#         weights.data = weights.data * cp.exp(1j * m * theta.data)
#         return weights

#     # Without AGF
#     def generate_reciprocal_weight_matrix(dists):
#         """ dists is the csr pairwise distance matrix between all cells in subdata/adata. """
#         # Create a copy of dists;
#         weights = cpx.csr_matrix(dists)
#         # Decay function to define weight as a function of distances; as a test use 1 / d; ideally use their scaled_gaussian fn
#         weights.data = 1 / weights.data
#         # Then replace infinties (where 1/0 occurred -> self distances)
#         weights.data[weights.data == cp.inf] = 0.0
#         # Row normalisation
#         weights = csr_row_normalisation(weights)
#         return weights 

#     def generate_scaled_gaussian_weight_matrix(dists):
#         # Extracting data and indptr from the CuPy sparse matrix. Inplace operation on dists;
#         weight_csr = cpx.csr_matrix(dists)
        
#         indptr, data = weight_csr.indptr, weight_csr.data
        
#         # Get realised k from the indptrs (k - 1; exclude self)
#         k = indptr[1] - indptr[0] - 1
        
#         # Get rows from indptrs
#         r = len(indptr) - 1
        
#         # Calculate median for each row; -> use index ptrs;
#         # We know K apriori, and that speeds up calculating median as its simply indexing;
#         # If K is even, then the left
#         if k % 2 == 0:
#             left_index = ((k-1) // 2) + 1
#             right_index = (-(-(k-1) // 2)) + 1
#             left_indices = indptr[:-1] + left_index
#             right_indices = indptr[:-1] + right_index
#             left_data = data[left_indices]
#             right_data = data[right_indices]
#             median_data = (left_data + right_data)/2
#         # If K is odd, then simply index data with median indices
#         else:
#             median_index = (k // 2) + 1
#             median_indices = indptr[:-1] + median_index # This gets all the median indices across all rows; so simply take data[median_indices]
#             median_data = data[median_indices]
        
#         # Compute scaled weights using broadcasting -> numerators;
#         weights = cp.exp(-((data.reshape((int(r), int(k+1))) / median_data.reshape(int(r), 1)) ** 2))
        
#         # weights = cpx.csr_matrix((weights.ravel(), weight_csr.indices, weight_csr.indptr))
#         # weights.setdiag(cp.float32(0))
        
#         # Set self distances to 0
#         weights[:,0] = 0.0
#         weights = cpx.csr_matrix((weights.ravel(), weight_csr.indices, weight_csr.indptr))
        
#         weights = csr_row_normalisation(weights)

#     # graph_out, distance_graph -> weights, dists
#     def generate_neighbor_matrix(weights, X):
#         # This generates gamma; which is simply W.C
#         if isinstance(X, np.ndarray):
#             X = cp.array(X)
#         if weights.shape[1] != X.shape[0]:
#             raise ValueError()
#         N = weights.dot(X)
#         return N 

#     def variance_balance(X, N):
#         """ Given the expression matrix; get the variance, then balance this 
#         with the variance of the neighbour expression matrix; """
#         import cuml
#         X_PCA = cuml.PCA(n_components=20)
#         X_PCA.fit_transform(cp.nan_to_num(cpxstats.zscore(X, axis=0), nan=0))
#         total_X_variance = np.sum(np.square(X_PCA.singular_values_))
        
#         N_PCA = cuml.PCA(n_components=20)
#         N_PCA.fit_transform(cp.nan_to_num(cpxstats.zscore(N, axis=0), nan=0))
#         total_N_variance = np.sum(np.square(N_PCA.singular_values_))
            
#         # Variance balancing here; sqrt(var(X)/var(N))
#         balance_factor = np.sqrt(total_X_variance/total_N_variance)
        
#         return balance_factor * N

#     def generate_banksy_matrix(
#             adata, 
#             spatial_neighbours_key="geom_neighbors", 
#             n_neighbors=15,
#             neighbour_influence=0.3,
#             m=0):
#         # If m>0 / perform AGF, then we need to compute multiple neighbours search 
#         # across many Ks

#         # Max k
#         max_k = n_neighbors * (m+1) 
#         # Compute KNNs;
#         knn = cuml.neighbors.NearestNeighbors(
#             n_neighbors = max_k + 1, # Due to results including self
#             algorithm = "brute",
#             metric = "euclidean",
#             output_type = "cupy"
#         )

#         locs = cp.array(adata.obsm["spatial"])
#         knn.fit(locs)
#         d, idx = knn.kneighbors(locs, two_pass_precision=False)

#         def _remove_self_distances(d, idx):
#             """ Remove self distances from the nearest neighbors output. """
#             if idx[0, 0] == 0:
#                 idx = cp.delete(idx, 0, axis=1)
#                 d = cp.delete(d, 0, axis=1)
#             else:  # Otherwise delete the _last_ column of d and idx
#                 idx = cp.delete(idx, -1, axis=1)
#                 d = cp.delete(d, -1, axis=1)
#             return d, idx

#         d, idx = _remove_self_distances(d, idx) # N x Knn shape
        
#         # Then loop through and subset d,idx for smaller K's
#         X = cp.array(adata.X.copy()) # Original Expression data
        
#         def weights_thetas_per_m(X, locs, d, idx, m):
#             # d, idx stores the parent KNN with the largest K
            
#             # For each m, subset the d, idx
#             # TODO:

#             # Convert into square adj matrix; sparse csr matrix format; N x N shape
#             d_flat = d.ravel()
#             indices = # ... # TODO: left off
#             indptr = # ... # TODO: left off
#             csr = # ... # TODO: left off 
#             dists = cpx.csr_matrix(
#                 (cp.array(csr.data), cp.array(csr.indices), cp.array(csr.indptr)), 
#                 shape=csr.shape, 
#                 dtype=cp.float64) # Technically knn

#             # Get the weights graph
#             W = generate_scaled_gaussian_weight_matrix(dists)

#             # Then get NBR matrix; (m == 0; non AGF)
#             N = generate_neighbor_matrix(W, X) # dot product

#             # Get the azimuthal/theta graph
#             if m > 0:
#                 T = generate_azimuth_matrix(locs, dists)
#                 # then weights is augmented;
#                 W = agf_transform(W, m, T)
#                 # TODO : implement sum operation for discrete FFT
#                 # Then take the weights matrix; absolute value it
#                 W_abs = W.copy()
#                 W_abs.data = cp.absolute(W_abs.data)
#                 N_abs = generate_neighbor_matrix(W_abs, X)
#                 N_mat = cp.zeros(X.shape,) # line 145

#                 # TODO: left off Line 146 onwards to work on

#         # Scale each feature matrix by lambda parameter; 
#         N_influence = 0.3
#         X *= cp.sqrt(1 - N_influence)
#         N *= cp.sqrt(N_influence)
#         B = cp.hstack((X, N)).get()

#         # and create the adata with these new set of features; banksy_matrix_to_adata;
#         var_nbrs = adata.var.copy()
#         var_nbrs.index += "_nbr"
#         nbr_bool = np.zeros((var_nbrs.shape[0] * 2,), dtype=bool)
#         nbr_bool[var_nbrs.shape[0]:] = True
#         var_combined = pd.concat([adata.var, var_nbrs])
#         var_combined["is_nbr"] = nbr_bool

#         return anndata.AnnData(B, obs=adata.obs, var=var_combined)

#     # def get_banksy_per_img(adata, spatial_n_neighbors, n_influence):
#     #     adata_by_img = []
#     #     rsc.utils.anndata_to_GPU(adata)
#     #     for img in adata.obs.Image.unique():
#     #         subdata = adata[adata.obs.Image == img]
#     #         rsc.pp.neighbors(subdata, use_rep="spatial", n_neighbors=spatial_n_neighbors, key_added=f"geometrical_{spatial_n_neighbors}nn")
#     #         adata_by_img.append(generate_banksy_matrix(subdata, f"geometrical_{spatial_n_neighbors}nn_distances", n_influence))
#     #     return adata_by_img
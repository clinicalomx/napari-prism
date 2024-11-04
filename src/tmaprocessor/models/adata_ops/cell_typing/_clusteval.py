# from here anndataeval
from anndata import AnnData
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

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
        self.quality_scores = adata.uns[searcher_name + "_quality_scores"]

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
            cluster labels array. Assert matched indexing of adata """
        labels = self.get_K(k)[r]
        labels.index = labels.index.astype(str)
        assert all(labels.index == self.adata.obs.index)
        return labels
    
    def get_annotated_cluster_labels(self):
        # Get the nicer dataframe version of obsm
        cluster_df = pd.DataFrame(self.cluster_labels)
        cluster_df.columns = self.cluster_label_map.values()
        cluster_df.columns = cluster_df.columns.set_names(("K", "R"))
        return cluster_df
    
    """ Below fall into metrics which technically need a 'ground' truth. 
        (But here we specify the ground truth as the cluster labels themselves)"""
    def between_model_score(self, score_function, k=None, **kwargs):
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
                
        if k is not None:
            pp_matrix = pp_matrix.loc[
                (pp_matrix.index.get_level_values("K") == k),
                (pp_matrix.columns.get_level_values("K") == k)
            ]
        return pp_matrix

    def adjusted_rand_index(self, k=None):
        """ For a N x K matrix, 
            Return a P x P array of adjusted rand indices for each P. """
        return self.between_model_score(
            self.ml.adjusted_rand_score, k)
    
    def normalized_mutual_info(self, k=None):
        """ For a N x K matrix, 
            Return a P x P array of normalized mutual info scores for each P. """
        return self.between_model_score(
            self.ml.normalized_mutual_info_score, k)
    
    def adjusted_mutual_info(self, k=None):
        """ For a N x K matrix, 
            Return a P x P array of adjusted mutual info scores for each parameter run P. """
        return self.between_model_score(
            self.ml.adjusted_mutual_info_score, k)
    
    def get_co_membership_matrix(self):
        cluster_labels = self.cluster_labels
        N = cluster_labels.shape[0]
        P = cluster_labels.shape[1]
        co_membership_matrix = np.zeros((N, N))

        # Update the co-membership matrix
        for p in range(P):
            for i in range(N):
                for j in range(N):
                    if cluster_labels[i, p] == cluster_labels[j, p]:
                        co_membership_matrix[i, j] += 1

        return co_membership_matrix
    
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
    
    def dotplot(self, k, r, layer=None):
        """ For a given K, R, plot a dotplot of the cluster labels. """
        adata_tmp = self.adata.copy()
        adata_tmp.obs["_tmp_lbl"] = list(self.get_K_R(k, r))
        adata_tmp.obs["_tmp_lbl"] = adata_tmp.obs["_tmp_lbl"].astype("category")
        sc.pl.dotplot(adata_tmp, adata_tmp.var_names, "_tmp_lbl", layer=layer)
        plt.show()
        del adata_tmp
    
    def matrixplot(self, k, r, layer=None):
        """ For a given K, R, plot a matrixplot of the cluster labels. """
        adata_tmp = self.adata.copy()
        adata_tmp.obs["_tmp_lbl"] = list(self.get_K_R(k, r))
        adata_tmp.obs["_tmp_lbl"] = adata_tmp.obs["_tmp_lbl"].astype("category")
        sc.pl.matrixplot(adata_tmp, adata_tmp.var_names, "_tmp_lbl", layer=layer)
        plt.show()
        del adata_tmp

    """ https://github.com/crazyhottommy/scclusteval
    We then sample without replacement a subset of the data set 
    (e.g. 80% of the cells in the full data set), and then repeat the clustering
    procedure on just this subset of data (so repeating all aspects of 
    clustering, including calling variable genes, calculating PCs, building the 
    neighbor graph, etc), and we do this n times."""
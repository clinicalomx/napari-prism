# from here anndataeval

from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData


class ClusteringSearchEvaluator:
    """Assess 'quality' of clusteriung results from a clustering search."""

    IMPLEMENTED_SEARCHERS = [
        "ScanpyClusteringSearch",
        "HybridPhenographSearch",
    ]

    def __init__(
        self,
        adata: AnnData,
        searcher_name: Literal[
            "ScanpyClusteringSearch", "HybridPhenographSearch"
        ],
        gpu: bool = False,
    ) -> None:
        """
        Args:
            adata (AnnData): Anndata object containing results from the
                clustering search.
            searcher_name (str): Name of the clustering searcher. Options:
                "ScanpyClusteringSearch", "HybridPhenographSearch".
            gpu (bool, optional): If available, attempts to use GPU for
                clustering. Defaults to False.
        """
        self.adata = adata
        self.searcher_name = searcher_name

        #: pd.DataFrame of labels from clustering search
        self.cluster_labels = adata.obsm[searcher_name + "_labels"]

        #: pd.DataFrame of quality scores from clustering search
        self.quality_scores = adata.uns[searcher_name + "_quality_scores"]

        if gpu:
            self._set_ml_backend("cuml")
        else:
            self._set_ml_backend("sklearn")

    def _set_ml_backend(self, backend: Literal["sklearn", "cuml"]) -> None:
        """Set the backend for computing machine learning metrics.

        Args:
            backend (str): Backend to use for computing machine learning
                metrics. Options: "sklearn", "cuml".
        """
        if backend == "sklearn":
            import sklearn.metrics as metrics
        elif backend == "cuml":
            import cuml.metrics.cluster as metrics  # type: ignore
        else:
            raise ValueError("Invalid backend. Options: sklearn, cuml")

        self.ml_backend = backend
        self.ml = metrics

    def get_K(self, k: int) -> pd.DataFrame:
        """Gets the clustering search run subset to a given `k`.

        Args:
            k: K parameter value for the number of neighbors computed.

        Returns:
            Subset pd.DataFrame containing columns of a given `k`.
        """
        keys_with_k = [
            x for x in self.cluster_labels.columns if x.split("_")[0] == str(k)
        ]

        cluster_df = self.cluster_labels[keys_with_k]
        cluster_df.columns = [x.split("_")[1] for x in cluster_df.columns]
        cluster_df.columns.name = "R"
        return cluster_df

    def get_K_R(self, k, r) -> pd.Series:
        """Returns the cluster labels from a given clustering run with a given
        `k` and `r`.

        Args:
            k: K parameter value for the number of neighbors computed.
            r: Graph clustering resolution value.

        Returns:
            pd.Series containing the cluster labels.
        """
        labels = self.get_K(k)[str(r)]
        labels.index = labels.index.astype(str)
        # assert all(labels.index == self.adata.obs.index)
        return labels

    def get_annotated_cluster_labels(self) -> pd.DataFrame:
        """Return the cluster label matrix with multi-indexed columns for `k`
        and `r`.

        Returns:
            pd.DataFrame with multi-indexed columns, where the first level is
            `k` and the second level is `r`.
        """
        # Get the nicer dataframe version of obsm
        cluster_df = self.cluster_labels.copy()
        cluster_df.columns = [
            tuple(k_r) for k_r in cluster_df.columns.str.split("_")
        ]
        cluster_df.columns = pd.MultiIndex.from_tuples(cluster_df.columns)
        cluster_df.columns = cluster_df.columns.set_names(("K", "R"))
        return cluster_df

    def between_model_score(
        self, score_function: callable, k: int | None = None, **kwargs
    ) -> pd.DataFrame:
        """Return a pairwise array of quality scores between every other
        clustering run.

        Args:
            score_function: Function to compute a score between the labels of
                two clustering runs.
            k: K parameter value for the number of neighbors computed. If not
                None, the pairwise array only computes scores for the given `k`.
            **kwargs: Additional keyword arguments to pass to the score
                function.

        Returns:
            pd.DataFrame of pairwise scores between clustering runs.
        """
        df = self.get_annotated_cluster_labels()
        p_len = df.shape[1]
        pp_matrix = pd.DataFrame(
            np.zeros((p_len, p_len)), index=df.columns, columns=df.columns
        )

        for i in df.columns:
            for j in df.columns:
                pp_matrix.loc[i, j] = score_function(
                    df.loc[:, i].values, df.loc[:, j].values, **kwargs
                )

        if k is not None:
            k = str(k)
            pp_matrix = pp_matrix.loc[
                (pp_matrix.index.get_level_values("K") == k),
                (pp_matrix.columns.get_level_values("K") == k),
            ]
        return pp_matrix

    def adjusted_rand_index(self, k=None):
        """Wrapper function"""
        return self.between_model_score(self.ml.adjusted_rand_score, k)

    def normalized_mutual_info(self, k=None):
        """Wrapper function"""
        return self.between_model_score(
            self.ml.normalized_mutual_info_score, k
        )

    def adjusted_mutual_info(self, k=None):
        """Wrapper function"""
        return self.between_model_score(self.ml.adjusted_mutual_info_score, k)

    def within_model_score(
        self, score_function: callable, k: int | None = None, **kwargs
    ) -> pd.DataFrame:
        """Return an array of quality scores within each clustering run.
        Usually, `score_function` computes a score that assesss how well the
        assigned cluster labels aggregate in some data space."""
        raise NotImplementedError()

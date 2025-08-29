import pytest
import scanpy as sc
from anndata import AnnData

from napari_prism.models.adata_ops.cell_typing._clusteval import (
    ClusteringSearchEvaluator,
)
from napari_prism.models.adata_ops.cell_typing._clustsearch import (
    HybridPhenographSearch,
)

KS = [3, 4]  # ints
RS = [0.5, 1.0]  # floats due to step
BASIS = "X_pca"


@pytest.fixture(scope="module")  # run once for tests here
def adata_post_phenosearch() -> AnnData:
    adata = sc.datasets.pbmc68k_reduced()
    # befores
    searcher = HybridPhenographSearch(
        knn="CPU", refiner="CPU", clusterer="CPU", clustering="leiden"
    )

    return searcher.parameter_search(adata, embedding_name=BASIS, ks=KS, rs=RS)


def test_hybrid_phenograph_search_cpu(adata_post_phenosearch):
    ADDED_OBSM_KEY = "HybridPhenographSearch_labels"
    ADDED_UNS_KEYS = ["HybridPhenographSearch_quality_scores", "param_grid"]
    EXPECTED_SHAPE = (adata_post_phenosearch.n_obs, len(KS) * len(RS))
    EXPECTED_COLUMNS = [str(k) + "_" + str(r) for k in KS for r in RS]

    # Check outputs
    assert ADDED_OBSM_KEY in adata_post_phenosearch.obsm
    assert all(k in adata_post_phenosearch.uns for k in ADDED_UNS_KEYS)

    assert adata_post_phenosearch.uns["param_grid"]["ks"] == KS
    assert adata_post_phenosearch.uns["param_grid"]["rs"] == RS

    labels = adata_post_phenosearch.obsm[ADDED_OBSM_KEY]
    assert labels.shape == EXPECTED_SHAPE
    assert all(labels.columns.values == EXPECTED_COLUMNS)
    assert all(labels.index == adata_post_phenosearch.obs.index)


def test_hybrid_phenograph_eval_cpu(adata_post_phenosearch):
    # tests
    evaluator = ClusteringSearchEvaluator(
        adata_post_phenosearch, "HybridPhenographSearch"
    )
    assert evaluator is not None
    assert hasattr(evaluator, "cluster_labels")
    assert hasattr(evaluator, "quality_scores")

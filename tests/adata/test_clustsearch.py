from napari_prism.models.adata_ops.cell_typing._clustsearch import (
    HybridPhenographSearch,
)


def test_hybrid_phenograph_search_cpu(adata):
    KS = [3, 4]  # ints
    RS = [0.5, 1.0]  # floats?
    ADDED_OBSM_KEYS = ["HybridPhenographSearch_labels"]
    ADDED_UNS_KEYS = ["HybridPhenographSearch_quality_scores", "param_grid"]
    EXPECTED_SHAPE = (adata.n_obs, len(KS) * len(RS))
    EXPECTED_COLUMNS = [str(k) + "_" + str(r) for k in KS for r in RS]

    searcher = HybridPhenographSearch(
        knn="CPU", refiner="CPU", clusterer="CPU", clustering="leiden"
    )

    out = searcher.parameter_search(  # noqa: F841
        adata, embedding_name="X_pca", ks=KS, rs=RS
    )

    # Check outputs
    assert all(k in adata.obsm for k in ADDED_OBSM_KEYS)
    assert all(k in adata.uns for k in ADDED_UNS_KEYS)

    assert adata.uns["param_grid"]["ks"] == KS
    assert adata.uns["param_grid"]["rs"] == RS

    labels = adata.obsm["HybridPhenographSearch_labels"]
    assert labels.shape == EXPECTED_SHAPE
    assert all(labels.columns.values == EXPECTED_COLUMNS)
    assert all(labels.index == adata.obs.index)

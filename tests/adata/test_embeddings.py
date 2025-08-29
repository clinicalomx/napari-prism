from unittest.mock import patch

import pytest

from napari_prism.models.adata_ops.cell_typing._embeddings import (
    _current_backend,
    harmony,
    pca,
    set_backend,
    tsne,
    umap,
)


def test_set_backend_gpu_fail():
    """Test setting backend to GPU/CPU."""
    # Testing environment should be a cpu only env;
    with pytest.raises(ImportError):
        set_backend("gpu")


def test_set_backend_unknown_fail():
    with pytest.raises(ValueError):
        set_backend("blah")


def test_set_backend_cpu_success():
    set_backend("cpu")
    assert _current_backend["module"] == "scanpy"


def test_set_backend_gpu_not_installed():
    """Test setting backend to GPU when rapids_singlecell is not installed."""
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError(
            "No module named 'rapids_singlecell'"
        )
        with pytest.raises(
            ImportError, match="rapids_singlecell not installed"
        ):
            set_backend("gpu")


def test_set_backend_invalid():
    """Test setting invalid backend."""
    with pytest.raises(
        ValueError, match="Invalid backend. Must be 'cpu' or 'gpu'"
    ):
        set_backend("invalid")


def test_pca(adata):
    TEST_N_COMPS = 11
    result = pca(adata, copy=True, n_comps=TEST_N_COMPS)
    # n_pcs is a kwargs in this case; can test succsseful passing
    assert "X_pca" in result.obsm
    nrows, ncols = result.obsm["X_pca"].shape
    assert nrows == adata.shape[0]
    assert ncols == TEST_N_COMPS


def test_umap(adata):
    """Test UMAP function."""
    TEST_N_COMPS = 2
    result = umap(adata, copy=True, min_dist=0.5, n_components=TEST_N_COMPS)
    assert "X_umap" in result.obsm
    nrows, ncols = result.obsm["X_umap"].shape
    assert nrows == adata.shape[0]
    assert ncols == TEST_N_COMPS


def test_tsne(adata):
    """Test t-SNE function."""
    TEST_N_COMPS = 2
    result = tsne(adata, copy=True, use_rep="X")  # by default 2D
    assert "X_tsne" in result.obsm
    nrows, ncols = result.obsm["X_tsne"].shape
    assert nrows == adata.shape[0]
    assert ncols == TEST_N_COMPS


def test_harmony_scanpy(adata):
    """Test Harmony function with scanpy backend."""
    # adata test, pbmc reduced has xpca, batch correct by phase just for tests
    adata = harmony(adata, copy=True, key="phase", basis="X_pca")
    print(adata)
    # added basis;
    ADDED_BASIS = "X_pca_harmony"
    assert ADDED_BASIS in adata.obsm
    assert adata.obsm["X_pca"].shape == adata.obsm["X_pca_harmony"].shape


def test_harmony_rapids(adata):
    """Test Harmony function with rapids backend.

    No GPU tests at the moment.
    """

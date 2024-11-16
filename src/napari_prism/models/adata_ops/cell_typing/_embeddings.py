""".tl module"""

import importlib
from collections.abc import Callable
from functools import wraps
from typing import Literal

import loguru
from anndata import AnnData

_current_backend = {"module": "scanpy"}
sc_backend = importlib.import_module(_current_backend["module"])


def set_backend(backend: Literal["cpu", "gpu"]) -> None:
    """Set the backend to use for processing."""
    global sc_backend
    if backend == "cpu":
        _current_backend["module"] = "scanpy"
        loguru.logger.info("Setting backend to CPU with scanpy")
    elif backend == "gpu":
        try:
            import rapids_singlecell  # noqa F401

            _current_backend["module"] = "rapids_singlecell"
            loguru.logger.info("Setting backend to GPU with rapids_singlecell")
        except ImportError as e:
            raise ImportError("rapids_singlecell not installed") from e
    else:
        raise ValueError("Invalid backend. Must be 'cpu' or 'gpu'")

    sc_backend = importlib.import_module(_current_backend["module"])


def with_current_backend(function: Callable) -> Callable:
    """Decorator to dynamically use current backend for scanpy-type functions.
    Also trims keyword arguments to only those accepted by the function.

    If GPU backend is set, then function handles moving data to GPU memory.
    After running the function, it always returns it back to CPU memory.

    Args:
        function: Scanpy or rapids_singlecell function to wrap.

    Returns:
        Wrapped function.
    """

    @wraps(function)
    def wrapper(adata, **kwargs):
        backend = _current_backend["module"]
        if backend == "rapids_singlecell":
            sc_backend.get.anndata_to_GPU(adata)

        function_kwargs = trim_kwargs(kwargs, function)
        adata = function(adata, **function_kwargs)

        if backend == "rapids_singlecell":
            sc_backend.get.anndata_to_CPU(adata)

        return adata

    return wrapper


def trim_kwargs(function_kwargs: dict, function: Callable) -> dict:
    """Trim function_kwargs to only those accepted by function.

    Args:
        function_kwargs: Keyword arguments to trim.
        function: Function to trim keyword arguments for.

    Returns:
        Trimmed keyword arguments.
    """
    return {
        k: v
        for k, v in function_kwargs.items()
        if k in function.__code__.co_varnames
    }


@with_current_backend
def pca(adata: AnnData, copy: bool = True, **kwargs) -> AnnData:
    """Perform principal components analysis. Wraps `sc.pp/tl.pca` or
    `rsc.pl/tl.pca`.

    Args:
        adata: Anndata object.
        copy: Return a copy instead of writing inplace.
        kwargs: Additional keyword arguments to pass to `pp/tl.pca`.

    Returns:
        Anndata object with PCA results in .obsm. If `copy` is False, modifies
        the AnnData object in place and returns None.
    """
    return sc_backend.tl.pca(adata, copy=copy, **kwargs)


@with_current_backend
def umap(adata: AnnData, copy: bool = True, **kwargs) -> AnnData:
    """Perform UMAP. Wraps `sc.pl/tl.umap` or `rsc.tl.umap`.

    Args:
        adata: Anndata object.
        copy: Return a copy instead of writing inplace.
        kwargs: Additional keyword arguments to pass to `tl.umap`.

    Returns:
        Anndata object with UMAP results in .obsm. If `copy` is False, modifies
        the AnnData object in place and returns None.
    """
    return sc_backend.tl.umap(adata, copy=copy, **kwargs)


@with_current_backend
def tsne(adata: AnnData, copy: bool = True, **kwargs) -> AnnData:
    """Perform t-SNE. Wraps `sc.pl/tl.tsne` or `rsc.tl.tsne`.

    Args:
        adata: Anndata object.
        copy: Return a copy instead of writing inplace.
        kwargs: Additional keyword arguments to pass to `tl.tsne`.

    Returns:
        Anndata object with t-SNE results in .obsm. If `copy` is False, modifies
        the AnnData object in place and returns None.
    """
    return sc_backend.tl.tsne(adata, copy=copy, **kwargs)


@with_current_backend
def harmony(adata: AnnData, copy: bool = True, **kwargs) -> AnnData:
    """Performs HarmonyPy batch correction. Wraps
    `sc.external.pp.harmony_integrate` or `rsc.pp.harmony_integrate`.

    Args:
        adata: Anndata object.
        copy: Return a copy instead of writing inplace.
        kwargs: Additional keyword arguments to pass to `pp.harmony_integrate`.

    Returns:
        Anndata object with Harmony results in .obsm. If `copy` is False,
        modifies the AnnData object in place and returns None.

    """
    if copy:
        adata = adata.copy()

    assert "key" in kwargs
    assert "basis" in kwargs

    key = kwargs.pop("key")
    basis = kwargs.pop("basis")
    adjusted_basis = f"{basis}_harmony"

    # In-place operation if rsc,
    sc_backend.pp.harmony_integrate(
        adata, key, basis=basis, adjusted_basis=adjusted_basis, **kwargs
    )

    if copy:
        return adata

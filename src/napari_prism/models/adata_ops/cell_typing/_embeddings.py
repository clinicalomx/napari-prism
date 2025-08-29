""".tl module. Backend (cpu/gpu) logic handled here."""

import importlib
import inspect
from collections.abc import Callable
from functools import wraps
from typing import Literal

import loguru
from anndata import AnnData

_current_backend = {"module": "scanpy"}
sc_backend = importlib.import_module(_current_backend["module"])


def set_backend(backend: Literal["cpu", "gpu"]) -> None:
    """
    Set the backend to use for processing. If GPU is selected, it will use
    `rapids_singlecell`. If CPU is selected, it will use `scanpy`.
    This function should be called before any other functions in this module
    are called.

    Args:
        backend: Backend to use. Must be either "cpu" or "gpu".

    """
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


def with_current_backend(target: Callable = None):
    """
    Decorator that:
      - If given a backend function, wraps it directly
      - If used as a decorator, wraps a custom function

    Provides:
      - backend-aware AnnData handling (GPU/CPU)
      - kwarg trimming based on function signature
      - copy=True is forced as the default, regardless of backend
    """

    sig = inspect.signature(target)

    @wraps(target)
    def wrapper(adata: AnnData, *args, **kwargs):
        backend = _current_backend.get("module")
        if backend not in ("scanpy", "rapids_singlecell"):
            raise RuntimeError(f"Unsupported backend: {backend}")

        if adata.is_view:
            adata = adata.copy()

        if backend == "rapids_singlecell":
            sc_backend.get.anndata_to_GPU(adata)

        # Enforce copy=True unless explicitly set
        if "copy" in sig.parameters and "copy" not in kwargs:
            kwargs["copy"] = True

        result = target(adata, *args, **kwargs)

        # Evaluate copy semantics
        copy_requested = kwargs.get("copy", True)
        adata_out = result if copy_requested else adata

        keep_on_gpu = kwargs.get("keep_on_gpu", False)
        if backend == "rapids_singlecell" and not keep_on_gpu:
            sc_backend.get.anndata_to_CPU(adata_out)

        return adata_out

    return wrapper


# pass through version,
pca = with_current_backend(sc_backend.tl.pca)
umap = with_current_backend(sc_backend.tl.umap)
tsne = with_current_backend(sc_backend.tl.tsne)


@with_current_backend
def harmony(adata: AnnData, copy: bool = True, **kwargs) -> AnnData:
    """
    Performs HarmonyPy batch correction. Wraps
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

    print(adata)
    assert "key" in kwargs
    assert "basis" in kwargs

    key = kwargs.pop("key")
    basis = kwargs.pop("basis")
    adjusted_basis = f"{basis}_harmony"

    # In-place operation if rsc,
    if _current_backend["module"] == "scanpy":
        sc_backend.external.pp.harmony_integrate(
            adata, key, basis=basis, adjusted_basis=adjusted_basis, **kwargs
        )
    else:  # rapids
        sc_backend.pp.harmony_integrate(
            adata, key, basis=basis, adjusted_basis=adjusted_basis, **kwargs
        )

    if copy:
        return adata

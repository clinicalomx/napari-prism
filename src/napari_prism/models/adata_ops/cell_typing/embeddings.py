""".tl module. Backend (cpu/gpu) logic handled here."""

from anndata import AnnData


def pca(adata, copy: bool = True, *args, **kwargs):
    backend = "cpu"
    if "backend" in kwargs:
        backend = kwargs["backend"]
        del kwargs["backend"]

    backend = backend.lower()
    assert backend in ("cpu", "gpu")
    if backend == "cpu":
        import scanpy as sc

        out = sc.tl.pca(adata, copy=copy, **kwargs)
    else:
        import rapids_singlecell as rsc

        rsc.get.anndata_to_GPU(adata)
        out = rsc.tl.pca(adata, copy=copy, **kwargs)
        if copy:
            rsc.get.anndata_to_CPU(out)
        else:
            rsc.get.anndata_to_CPU(adata)

    return out


def umap(adata, copy: bool = True, *args, **kwargs):
    backend = "cpu"
    if "backend" in kwargs:
        backend = kwargs["backend"]
        del kwargs["backend"]

    backend = backend.lower()
    assert backend in ("cpu", "gpu")
    if backend == "cpu":
        import scanpy as sc

        out = sc.tl.umap(adata, copy=copy, **kwargs)
    else:
        import rapids_singlecell as rsc

        rsc.get.anndata_to_GPU(adata)
        out = rsc.tl.umap(adata, copy=copy, **kwargs)
        if copy:
            rsc.get.anndata_to_CPU(out)
        else:
            rsc.get.anndata_to_CPU(adata)

    return out


def tsne(adata, copy: bool = True, *args, **kwargs):
    backend = "cpu"
    if "backend" in kwargs:
        backend = kwargs["backend"]
        del kwargs["backend"]

    backend = backend.lower()
    assert backend in ("cpu", "gpu")
    if backend == "cpu":
        import scanpy as sc

        out = sc.tl.tsne(adata, copy=copy, **kwargs)
    else:
        import rapids_singlecell as rsc

        rsc.get.anndata_to_GPU(adata)
        out = rsc.tl.tsne(adata, copy=copy, **kwargs)
        if copy:
            rsc.get.anndata_to_CPU(out)
        else:
            rsc.get.anndata_to_CPU(adata)

    return out


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

    backend = "cpu"
    if "backend" in kwargs:
        backend = kwargs["backend"]
        del kwargs["backend"]
    assert "key" in kwargs
    assert "basis" in kwargs

    key = kwargs.pop("key")
    basis = kwargs.pop("basis")
    adjusted_basis = f"{basis}_harmony"

    # In-place operation if rsc,
    if backend == "cpu":
        import scanpy as sc

        sc.external.pp.harmony_integrate(
            adata, key, basis=basis, adjusted_basis=adjusted_basis, **kwargs
        )
    else:
        import rapids_singlecell as rsc

        rsc.pp.harmony_integrate(
            adata, key, basis=basis, adjusted_basis=adjusted_basis, **kwargs
        )

    if copy:
        return adata

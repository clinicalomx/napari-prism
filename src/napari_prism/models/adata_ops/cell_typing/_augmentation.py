# # Banksy
# # Set backends
import anndata as ad
import numpy as np
import pandas as pd
from anndata import AnnData

# import cupyx.scipy.stats as cpxstats
# import cupy as cp
# import rapids_singlecell as rsc
# import cupyx.scipy.sparse as cpx
# import cuml


def add_obs_as_var(
    adata: AnnData, obs_columns: list[str], layer: str | None = None
) -> AnnData:
    """Adds .obs features as .var features to an AnnData.

    Args:
        adata: Anndata object.
        obs_columns: .obs columns to add as .var features.
        layer (str, optional): Layer to add .obs features to. Defaults to None.

    Returns:
        New Anndata object with added .var features.
    """

    X = adata.layers[layer] if layer is not None else adata.X

    assert all(x in adata.obs.columns for x in obs_columns)

    if isinstance(obs_columns, str):
        obs_columns = [obs_columns]

    # Check if obs_columns is already in the anndata; subset to those that
    # are new
    obs_columns = [x for x in obs_columns if x not in adata.var_names]

    aug_X = np.hstack(
        (X, adata.obs[obs_columns].values.reshape(-1, len(obs_columns)))
    )
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


def subset_adata_by_var(adata: AnnData, var_subset: list[str]) -> AnnData:
    """Subsets an AnnData object by var names.

    Args:
        adata: Anndata object.
        var_subset: .var names to subset by.

    Returns:
        View of the subsetted AnnData object.
    """

    assert all(x in adata.var_names for x in var_subset)
    return adata[:, var_subset]


def subset_adata_by_obs_category(
    adata: AnnData, obs_key: str, obs_subset: list[str]
) -> AnnData:
    """Subsets an AnnData object by obs category or categories.

    Args:
        adata: Anndata object.
        obs_subset: .obs names to subset by.

    Returns:
        View of the subsetted AnnData object.
    """

    assert all(x in adata.obs[obs_key].unique() for x in obs_subset)
    return adata[adata.obs[obs_key].isin(obs_subset)]

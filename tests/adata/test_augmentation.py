from typing import Any

from napari_prism.models.adata_ops.cell_typing._augmentation import (
    add_obs_as_var,
    subset_adata_by_obs_category,
    subset_adata_by_var,
)


def test_add_obs_as_var(adata: Any) -> None:
    obs_columns = ["S_score", "G2M_score"]
    new_vars = ["S_score_var", "G2M_score_var"]  # Relabelled
    augmented_adata = add_obs_as_var(adata, obs_columns)
    assert all(col in augmented_adata.var_names for col in new_vars)
    assert augmented_adata.n_vars == 767  # 765 original genes + the scores


def test_subset_adata_by_var(adata: Any) -> None:
    var_subset = ["HES4", "SUMO3", "ITGB2"]
    augmented_adata = subset_adata_by_var(adata, var_subset)
    assert set(augmented_adata.var_names) == set(var_subset)
    assert augmented_adata.n_vars == 3


def test_subset_adata_by_obs_category(adata: Any) -> None:
    KEEP_CATEGORIES = ["G1", "S"]
    CATEGORY_COUNTS = 683
    CATEGORY_KEY = "phase"
    adata_subset = subset_adata_by_obs_category(
        adata, obs_key=CATEGORY_KEY, obs_subset=KEEP_CATEGORIES
    )
    assert adata_subset.shape[0] == CATEGORY_COUNTS
    assert set(adata_subset.obs["phase"].unique()) == set(KEEP_CATEGORIES)

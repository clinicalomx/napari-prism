from typing import Any
import numpy as np
import pytest
# from typing import List 
# import anndata import AnnData

# def test_collect_child_adatas() -> List[AnnData]:
#     # Should return a list of its DIRECT children adatas;
#     pass

def test_inherit_children_obs(adata_tree_widget_populated: Any):
    EXPECTED_NEW_COLS = [
        "100->new",
        "50->50_HES4_TNFRSF4->new", 
        "50->50_SSU72_PARK7->new"
        ]
    root_node = adata_tree_widget_populated.topLevelItem(0)
    root_node.inherit_children_obs()

    root_obs = root_node.adata.obs
    print(root_obs.columns)
    assert all(c in root_obs.columns for c in EXPECTED_NEW_COLS)

    for c in EXPECTED_NEW_COLS:
        vc = root_obs[c].value_counts(dropna=False)
        assert np.nan in vc
        assert vc[np.nan] == 650 # pbmcreduced -> 700, - 50 assigned
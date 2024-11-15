from typing import Any

import pytest
import scanpy as sc
from anndata import AnnData
from qtpy.QtWidgets import QTreeWidget

from napari_prism.models.adata_ops.cell_typing._subsetter import AnnDataNodeQT


@pytest.fixture
def adata() -> AnnData:
    return sc.datasets.pbmc68k_reduced()


# @pytest.fixture
# def adata_with_phenograph_search(adata: Any) -> AnnData:
#     LABELS_MATRIX = np.


@pytest.fixture
def adata_tree_widget_populated(qtbot, adata: Any) -> QTreeWidget:
    """
    Root
     |------------------|
     |                  |
    100                 50
                        |-----------------------|
                        |                       |
                50_HES4_TNFRSF4          50_SSU72_PARK7
                                                |
                                            TERMINAL

    """
    tree_widget = QTreeWidget()
    HEADERS = ("AnnData Subset", "Properties")
    tree_widget.setColumnCount(len(HEADERS))
    tree_widget.setHeaderLabels(HEADERS)
    root_node = AnnDataNodeQT(adata, None, "Root", tree_widget)
    tree_widget.setCurrentItem(root_node)

    # Add additionals
    sub_obs_terminal = adata[50:100]
    sub_obs_terminal.obs["100_annotation"] = 1
    _ = AnnDataNodeQT(
        sub_obs_terminal, None, "100", parent=tree_widget.currentItem()
    )

    sub_obs = adata[:50]
    sub = AnnDataNodeQT(
        sub_obs, None, "50", parent=tree_widget.currentItem()
    )  # Set to current
    tree_widget.setCurrentItem(sub)

    # TODO Add new obs
    sub_obs_var1 = sub_obs[:, :2]  # HES4, TNFRSF4
    sub_obs_var1.obs["HES4_annotation"] = 1
    _ = AnnDataNodeQT(
        sub_obs_var1, None, "50_HES4_TNFRSF4", parent=tree_widget.currentItem()
    )

    # TODO Add new obs
    sub_obs_var2 = sub_obs[:, 2:4]  # SSU72, PARK7
    sub_obs_var2.obs["SSU72_annotation"] = 2
    sub2v2 = AnnDataNodeQT(
        sub_obs_var2, None, "50_SSU72_PARK7", parent=tree_widget.currentItem()
    )

    tree_widget.setCurrentItem(sub2v2)
    sub_obs_var2_sub = sub_obs_var2[:, 1:]
    sub_obs_var2_sub.obs["TERMINAL_annotation"] = (
        21  # This 'new' column is also in above..
    )
    _ = AnnDataNodeQT(
        sub_obs_var2_sub, None, "TERMINAL", parent=tree_widget.currentItem()
    )

    return tree_widget

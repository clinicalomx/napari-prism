import random
from typing import Any

import pytest
import scanpy as sc
import squidpy as sq
from anndata import AnnData
from qtpy.QtWidgets import QTreeWidget

from napari_prism.models.adata_ops.cell_typing.tree import AnnDataNodeQT
from napari_prism.models.adata_ops.spatial_analysis.simulation.poisson import (
    simulate_poisson_roi_data_from_adata,
)


@pytest.fixture()
def adata() -> AnnData:
    return sc.datasets.pbmc68k_reduced()


@pytest.fixture()
def adata_spatial(adata) -> AnnData:
    """
    Generate a poisson process over the anndata for 'fake' spatial coords.

    Simulate >1 sample to mimic ROI-type spatial datasets like TMAs.
    """
    sp_key = "spatial"
    ncells = adata.shape[0]
    adata.obs["library"] = [random.choice(["s1", "s2"]) for _ in range(ncells)]
    adata_with_fake_coord = simulate_poisson_roi_data_from_adata(
        adata, spatial_key_added=sp_key, library_key="library", roi_radius=1000
    )
    return adata_with_fake_coord


@pytest.fixture()
def adata_spatial_graph(adata_spatial) -> AnnData:
    """
    Generate a sparse matrix representation of spatial connectedness with the
    fake AnnData coordinates, stored in .obsp.

    Spatial connectedness here is assumed to be a K=10 KNN for simplicity.
    """
    sq.gr.spatial_neighbors(
        adata_spatial,
        spatial_key="spatial",
        coord_type="generic",
        n_neighs=10,
    )
    return adata_spatial


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

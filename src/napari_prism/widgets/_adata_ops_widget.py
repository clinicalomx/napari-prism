from pathlib import Path

import napari
import spatialdata as sd
from magicgui.widgets import ComboBox
from napari.utils.events import EmitterGroup
from qtpy.QtWidgets import QTableWidget, QTabWidget, QVBoxLayout, QWidget

from napari_prism.widgets.adata_ops._base_widgets import AnnDataOperatorWidget
from napari_prism.widgets.adata_ops._cell_typing_widgets import (
    AnnDataTreeWidget,
    AugmentationWidget,
    ClusterAnnotatorWidget,
    ClusterAssessmentWidget,
    ClusterSearchWidget,
    PreprocessingWidget,
    SubclusteringWidget,
)
from napari_prism.widgets.adata_ops._spatial_analysis_widgets import (
    GraphBuilderWidget,
    NolanWidget,
    ProximityDensityWidget,
)


class CellTypingTab(QTabWidget):
    """UI tabs."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata, tree):
        super().__init__()
        self.viewer = viewer
        self.tree = tree

        self.augmentation = AugmentationWidget(self.viewer, adata)
        self.augmentation.max_height = 900
        self.augmentation.events.augment_created.connect(
            lambda x: self.tree.add_node_to_current(
                x.adata, node_label=x.label
            )
        )
        self.addTab(self.augmentation.native, "Augmentation")

        self.preprocessor = PreprocessingWidget(self.viewer, adata)
        self.preprocessor.max_height = 900
        self.preprocessor.events.augment_created.connect(
            lambda x: self.tree.add_node_to_current(
                x.adata, node_label=x.label
            )
        )
        self.preprocessor.events.adata_changed.connect(
            lambda x: self.tree.update_model(x.adata)
        )
        self.addTab(self.preprocessor.native, "Preprocessing")

        self.clustering_searcher = ClusterSearchWidget(self.viewer, adata)
        self.clustering_searcher.max_height = 400
        self.clustering_searcher.events.adata_changed.connect(
            lambda x: self.tree.update_model(x.adata)
        )
        self.addTab(self.clustering_searcher.native, "Clustering Search")

        self.cluster_assessment = ClusterAssessmentWidget(self.viewer, adata)
        self.cluster_assessment.max_height = 900
        self.cluster_assessment.events.adata_changed.connect(
            lambda x: self.tree.update_model(x.adata)
        )
        self.addTab(self.cluster_assessment.native, "Assess Cluster Runs")

        self.cluster_annotator = ClusterAnnotatorWidget(self.viewer, adata)
        self.cluster_annotator.max_height = 900
        self.cluster_annotator.events.adata_changed.connect(
            lambda x: self.tree.update_model(x.adata)
        )
        self.addTab(self.cluster_annotator.native, "Visualise Clusters")

        # Needs root access
        self.subclusterer = SubclusteringWidget(self.viewer, adata)
        self.subclusterer.max_height = 700
        self.subclusterer.events.subcluster_created.connect(
            lambda x: self.tree.add_restored_node_to_current(
                x.adata_indices, node_label=x.label
            )
        )
        self.addTab(self.subclusterer.native, "Subclusterer")


class SpatialAnalysisTab(QTabWidget):
    """Spatial Analysis classes; 1) Squidpy Wrapper, 2) General Wrapper"""

    def __init__(self, viewer: "napari.viewer.Viewer", adata, tree):
        super().__init__()
        self.viewer = viewer
        self.tree = tree

        self.graph_builder = GraphBuilderWidget(self.viewer, adata)
        self.graph_builder.max_height = 400
        self.graph_builder.events.adata_changed.connect(
            lambda x: self.tree.update_model(x.adata)
        )
        self.addTab(self.graph_builder.native, "Build Graph")

        self.nolan_cn = NolanWidget(self.viewer, adata)
        self.nolan_cn.max_height = 400
        self.nolan_cn.events.adata_changed.connect(
            lambda x: self.tree.update_model(x.adata)
        )
        self.addTab(self.nolan_cn, "Cellular Neighborhoods")

        self.proximity_density = ProximityDensityWidget(self.viewer, adata)
        self.proximity_density.max_height = 400
        self.proximity_density.events.adata_changed.connect(
            lambda x: self.tree.update_model(x.adata)
        )
        self.addTab(self.proximity_density, "Proximity Density")


class FeatureModellingTab(QTableWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata, tree):
        super().__init__()
        self.viewer = viewer
        self.tree = tree


class AnnDataAnalysisParentWidget(QWidget):
    """UI tabs.

    This acts like a viewer-model.

    NOTE: FUTURE: Consider replacing or extending this with napari-spatialdata's
    DataModel, or adding it as a parameter.
    """

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.meta_adata = None
        self.meta_sdata = None
        self.events = EmitterGroup(
            source=self, meta_sdata_changed=None, meta_adata_changed=None
        )

        # If initial selection is valid, update
        init_selected = viewer.layers.selection.active
        if init_selected is not None and "sdata" in init_selected.metadata:
            self.update_sdata_model()

        self.viewer.layers.selection.events.changed.connect(
            self.update_sdata_model
        )

        # self.events.meta_sdata_changed.connect(
        #     lambda x: AnnDataOperatorWidget.update_sdata_all_operators(x.sdata)
        # )

        self.layout = QVBoxLayout()

        self._adata_selection = ComboBox(
            name="LayersWithContainedAdata",
            choices=self.get_adata_in_sdata,
            label="Select a contained adata",
        )
        self._adata_selection.scrollable = True
        self._adata_selection.changed.connect(self.update_adata_model)
        self.layout.addWidget(self._adata_selection.native)

        # Parent Data Manager; Hold the memory reference to adatas in this class
        # On creation, empty
        self.tree = AnnDataTreeWidget(
            self.viewer, self.meta_adata, self.meta_sdata
        )
        self.tree.min_height = 175
        self.tree.max_height = 600
        self.layout.addWidget(self.tree.native)

        # Set sdata in tree before reseting the choices
        self.events.meta_sdata_changed.connect(
            lambda x: self.tree.set_sdata(x.sdata)
        )
        self.events.meta_sdata_changed.connect(self.refresh_adata_choices)
        # When the hotspot changes; update the tree
        self.events.meta_adata_changed.connect(
            lambda x: self.tree.create_model(
                adata=x.adata, table_path=x.table_path
            )
        )  # Create new tree

        # When adata changes, update all operators
        self.tree.events.adata_created.connect(
            lambda x: AnnDataOperatorWidget.create_model_all_operators(x.adata)
        )

        self.tree.events.adata_created.connect(
            lambda: self._adata_selection.reset_choices()
        )

        self.tree.events.adata_changed.connect(
            lambda x: AnnDataOperatorWidget.update_model_all_operators(x.adata)
        )

        # When a new node is added, a new table is added on disk -> Refresh the
        # Spatialdata widgets to see this new table from disk
        self.tree.events.node_changed.connect(
            lambda x: self.refresh_sdata_widget_choices(
                table_to_add=x.table_to_add,
                table_to_remove=x.table_to_remove,
            )
        )

        # Hotdesk Adata
        adata = self.tree.adata

        self.tabs = QTabWidget()

        self.cell_typing_tab = CellTypingTab(viewer, adata, self.tree)

        self.spatial_analysis_tab = SpatialAnalysisTab(
            viewer, adata, self.tree
        )

        self.feature_modelling_tab = FeatureModellingTab(
            viewer, adata, self.tree
        )

        self.tabs.addTab(self.cell_typing_tab, "Cell Typing")
        self.tabs.addTab(self.spatial_analysis_tab, "Spatial Analysis")
        self.tabs.addTab(self.feature_modelling_tab, "Feature Modelling")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # init
        if self._adata_selection.value is not None:
            self.update_adata_model()

    def get_adata_in_sdata(self, widget=None):
        if self.meta_sdata is not None:
            return list(self.meta_sdata.tables.keys())
        else:
            return []

    def get_layers_with_valid_contained_sdata(self, widget=None):
        # Reference to the sdata in ithe main mutliscale image / labels
        # TODO: change to labels input
        # similar to 'Tables annotating layers' from napari-spatialdata
        return [
            x.name
            for x in self.viewer.layers
            # if isinstance(
            #     x.data, napari.layers._multiscale_data.MultiScaleData
            # )
            if isinstance(x.data, napari.layers.Labels)
            and "sdata" in x.metadata
            and x.metadata["sdata"] is not None
            and x.metadata["sdata"].is_backed()
            and "adata" in x.metadata
            and x.metadata["adata"] is not None
            and x.metadata["adata"].shape[0]
            > 0  # and isinstance(l, napari.layers.Labels)
        ]

    def get_layers_with_contained_adata(self, widget=None):
        layers = [
            x.name
            for x in self.viewer.layers
            if "adata" in x.metadata
            and x.metadata["adata"] is not None
            and x.metadata["adata"].shape[0]
            > 0  # and isinstance(l, napari.layers.Labels)
        ]

        if layers is None:
            raise AttributeError("No layers with contained adata found.")

        return layers

    def is_valid_selection(self, selected):
        """Determines if the selected layer is a valid parent layer from which
        to retrieve SpatialData AND AnnData objects from.
        """
        # TODO: optimise below, maybe use walrus, but less readable
        return (
            selected is not None
            # and isinstance(
            #     selected.data, napari.layers._multiscale_data.MultiScaleData
            # )
            and isinstance(selected, napari.layers.Labels)
            and "sdata" in selected.metadata
            and selected.metadata["sdata"] is not None
            and selected.metadata["sdata"].is_backed()
            and "adata" in selected.metadata
            and selected.metadata["adata"] is not None
            and "table_names" in selected.metadata
            and len(selected.metadata["table_names"]) > 0
        )

    def update_sdata_model(self):
        """NOTE: Consider using napari-spatialdata's DataModel."""
        selected = self.viewer.layers.selection.active
        sdata = None
        if self.is_valid_selection(selected):
            sdata = selected.metadata["sdata"]

        # If we have a new sdata, update
        if (
            sdata is not None
            and self.meta_sdata is None
            or self.meta_sdata is not sdata
        ):
            self.meta_sdata = sdata
            self.events.meta_sdata_changed(sdata=self.meta_sdata)

    def update_adata_model(self):
        selection = self._adata_selection.value
        self.meta_adata = self.meta_sdata[selection]
        # Enforce string indices for all parsed AnnDatas
        self.meta_adata.obs.index = self.meta_adata.obs.index.astype(str)
        table_path = Path(self.meta_sdata.path, "tables", selection)
        self.events.meta_adata_changed(
            adata=self.meta_adata, table_path=table_path
        )

    def refresh_adata_choices(self):
        self._adata_selection.reset_choices()

    def get_sdata_widget(self):
        # NOTE: private API, temp solution
        # track: https://github.com/scverse/napari-spatialdata/issues/313
        return self.viewer.window._dock_widgets["SpatialData"].widget()

    def get_sdata_view_widget(self):
        # NOTE: private API, temp solution
        # track: https://github.com/scverse/napari-spatialdata/issues/313
        return self.viewer.window._dock_widgets[
            "View (napari-spatialdata)"
        ].widget()

    def get_sdata_scatter_widget(self):
        # NOTE: private API, temp solution
        # track: https://github.com/scverse/napari-spatialdata/issues/313
        return self.viewer.window._dock_widgets[
            "Scatter (napari-spatialdata)"
        ].widget()

    def reload_sdata(self):
        self.meta_sdata = sd.read_zarr(self.meta_sdata.path)
        for layer in self.viewer.layers:
            if (
                "sdata" in layer.metadata
                and layer.metadata["sdata"] is not None
                and layer.metadata["sdata"].path == self.meta_sdata.path
            ):
                layer.metadata["sdata"] = self.meta_sdata

    def refresh_sdata_widget_choices(
        self,
        table_to_add: str | None = None,
        table_to_remove: str | None = None,
    ):
        # # Reload sdata
        # self.reload_sdata()

        # Add to current layer as table_names;
        for layer in self.viewer.layers:
            if (
                "sdata" in layer.metadata
                and "table_names" in layer.metadata
                and layer.metadata["table_names"] is not None
            ):
                if (
                    table_to_add
                    and table_to_add not in layer.metadata["table_names"]
                ):
                    layer.metadata["table_names"].append(table_to_add)

                if (
                    table_to_remove
                    and table_to_remove in layer.metadata["table_names"]
                ):
                    layer.metadata["table_names"].remove(table_to_remove)

        # Refresh the widgets to see the new table

        # View (napari-spatialdata)
        self.get_sdata_view_widget()._on_layer_update()

        # NOTE: hacky temp solution trigger a fake event to update the
        # scatter widget..
        class FakeEvent:
            def __init__(self, source, active):
                self.source = source
                self.active = active

        fake_event = FakeEvent(
            source=self.viewer.layers.selection.events.changed.source,
            active=self.viewer.layers.selection.active,
        )
        # Scatter (napari-spatialdata)
        self.get_sdata_scatter_widget()._on_selection(fake_event)

import napari
from magicgui.widgets import ComboBox
from napari.utils.events import EmitterGroup
from qtpy.QtWidgets import QTableWidget, QTabWidget, QVBoxLayout, QWidget

from napari_prism.widgets._widget_utils import (
    get_selected_layer,
    make_unique_sdata_element_name,
)
from napari_prism.widgets.adata_ops._base_widgets import AnnDataOperatorWidget
from napari_prism.widgets.adata_ops._cell_typing_widgets import (
    AnnDataSubsetterWidget,
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

    def __init__(self, viewer: "napari.viewer.Viewer", adata, subsetter):
        super().__init__()
        self.viewer = viewer
        self.subsetter = subsetter

        self.augmentation = AugmentationWidget(self.viewer, adata)
        self.augmentation.max_height = 400
        self.augmentation.events.augment_created.connect(
            lambda x: self.subsetter.add_node_to_current(
                x.value[0], node_label=x.value[1]
            )
        )
        self.addTab(self.augmentation.native, "Augmentation")

        self.preprocessor = PreprocessingWidget(self.viewer, adata)
        self.preprocessor.max_height = 900
        self.preprocessor.events.augment_created.connect(
            lambda x: self.subsetter.add_node_to_current(
                x.value[0], node_label=x.value[1]
            )
        )
        self.addTab(self.preprocessor.native, "Preprocessing")

        self.clustering_searcher = ClusterSearchWidget(self.viewer, adata)
        self.clustering_searcher.max_height = 400
        self.addTab(self.clustering_searcher.native, "Clustering Search")

        self.cluster_assessment = ClusterAssessmentWidget(self.viewer, adata)
        self.cluster_assessment.max_height = 700
        self.addTab(self.cluster_assessment.native, "Assess Cluster Runs")

        self.cluster_annotator = ClusterAnnotatorWidget(self.viewer, adata)
        self.cluster_annotator.max_height = 900
        self.addTab(self.cluster_annotator.native, "Annotate Clusters")

        # Needs root access
        self.subclusterer = SubclusteringWidget(self.viewer, adata)
        self.subclusterer.max_height = 700
        self.subclusterer.events.subcluster_created.connect(
            lambda x: self.subsetter.add_node_to_current(
                x.value[0], node_label=x.value[1]
            )
        )
        self.addTab(self.subclusterer.native, "Subclusterer")


class SpatialAnalysisTab(QTabWidget):
    """Spatial Analysis classes; 1) Squidpy Wrapper, 2) General Wrapper"""

    def __init__(self, viewer: "napari.viewer.Viewer", adata, subsetter):
        super().__init__()
        self.viewer = viewer
        self.subsetter = subsetter

        self.graph_builder = GraphBuilderWidget(self.viewer, adata)
        self.graph_builder.max_height = 400
        self.addTab(self.graph_builder.native, "Build Graph")

        self.nolan_cn = NolanWidget(self.viewer, adata)
        self.nolan_cn.max_height = 400
        self.addTab(self.nolan_cn, "Cellular Neighborhoods")

        self.proximity_density = ProximityDensityWidget(self.viewer, adata)
        self.proximity_density.max_height = 400
        self.addTab(self.proximity_density, "Proximity Density")


class FeatureModellingTab(QTableWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata, subsetter):
        super().__init__()
        self.viewer = viewer
        self.subsetter = subsetter


class AnnDataAnalysisParentWidget(QWidget):
    """UI tabs."""

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

        # self.viewer.layers.events.inserted.connect(self.refresh_choices_from_image)
        # self.viewer = viewer

        # Maintain adata state of current selected layer
        # self.loader = LoaderWidget(self.viewer)
        # self.loader.max_height = 200
        # self.addTab(self.loader.native, "Loader")
        self.viewer.layers.selection.events.changed.connect(
            self.update_sdata_model
        )

        self.events.meta_sdata_changed.connect(self.refresh_adata_choices)
        self.events.meta_sdata_changed.connect(
            lambda x: AnnDataOperatorWidget.update_sdata_all_operators(x.value)
        )

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
        self.subsetter = AnnDataSubsetterWidget(self.viewer, None)
        self.subsetter.min_height = 120
        self.subsetter.max_height = 500
        self.layout.addWidget(self.subsetter.native)
        # When the hotspot changes; update the tree
        self.events.meta_adata_changed.connect(
            lambda x: self.subsetter.create_model(x.value)
        )  # Create new tree

        # When adata changes, update all operators
        self.subsetter.events.adata_created.connect(
            lambda x: AnnDataOperatorWidget.create_model_all_operators(x.value)
        )

        self.subsetter.events.adata_changed.connect(
            lambda x: AnnDataOperatorWidget.update_model_all_operators(x.value)
        )

        self.subsetter.events.adata_saved.connect(
            lambda x: self.save_adata_to_sdata(x.value)
        )

        # Hotdesk Adata
        adata = self.subsetter.adata

        self.tabs = QTabWidget()

        self.cell_typing_tab = CellTypingTab(viewer, adata, self.subsetter)

        self.spatial_analysis_tab = SpatialAnalysisTab(
            viewer, adata, self.subsetter
        )

        self.feature_modelling_tab = FeatureModellingTab(
            viewer, adata, self.subsetter
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
        # Reference to the sdata in ithe main mutliscale image
        return [
            x.name
            for x in self.viewer.layers
            if isinstance(
                x.data, napari.layers._multiscale_data.MultiScaleData
            )
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
        return (
            selected is not None
            and isinstance(
                selected.data, napari.layers._multiscale_data.MultiScaleData
            )
            and "sdata" in selected.metadata
            and selected.metadata["sdata"] is not None
            and selected.metadata["sdata"].is_backed()
        )

    def update_sdata_model(self):
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
            self.events.meta_sdata_changed(value=self.meta_sdata)

    def update_adata_model(self):
        selection = self._adata_selection.value
        self.meta_adata = self.meta_sdata[selection]
        self.events.meta_adata_changed(value=self.meta_adata)

    def refresh_adata_choices(self):
        self._adata_selection.reset_choices()

    def get_sdata_widget(self):
        # NOTE: private API, temp solution
        # track: https://github.com/scverse/napari-spatialdata/issues/313
        return self.viewer.window._dock_widgets["SpatialData"].widget()

    def save_adata_to_sdata(self, new_adata):
        selection = self._adata_selection.value
        node_name = self.subsetter.adata_tree_widget.currentItem().text(0)
        if node_name == "Root":
            save_name = f"{selection}_new"
        else:
            save_name = f"{selection}_{node_name}"

        # meta_adata = self.meta_sdata[selection]
        # Dont overwrite, track original?
        new_selection_label = make_unique_sdata_element_name(
            self.meta_sdata, save_name
        )
        # Can parse AnnData directly as it will inherit the attrs from uns
        self.meta_sdata[new_selection_label] = new_adata
        self.meta_sdata.write_element(new_selection_label)
        self._adata_selection.reset_choices()
        labels_name = new_adata.uns["spatialdata_attrs"]["region"]
        # time.sleep(0.1)
        if labels_name in self.viewer.layers:
            labels = get_selected_layer(self.viewer, labels_name)
            self.viewer.layers.remove(labels)
            sdata_widget = self.get_sdata_widget()
            sdata_widget._onClick(text=labels.name)

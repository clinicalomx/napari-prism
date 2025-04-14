import napari
from qtpy.QtWidgets import QTabWidget

from napari_prism.models.tma_ops._tma_image import (
    TMADearrayer,
    TMAMasker,
    TMAMeasurer,
    TMASegmenter,
)
from napari_prism.widgets.tma_ops._tma_image_widgets import (
    TMADearrayerNapariWidget,
    TMAMaskerNapariWidget,
    TMAMeasurerNapariWidget,
    TMASegmenterNapariWidget,
    UtilsNapariWidget,
)


class TMAImageAnalysisParentWidget(QTabWidget):
    """UI tabs."""

    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.dearrayer = None
        self.segmenter = None

        # bind viewer selection events such that labels if they do get imported
        # with annotations
        self.viewer.layers.selection.events.changed.connect(
            self.on_selection_change
        )

        # self.general = GeneralMSNapariWidget(self._viewer)
        # self.addTab(self.general.native, "Other")
        init_selected = viewer.layers.selection.active

        init_model_masker = None
        init_model_dearrayer = None
        init_model_segmenter = None
        init_model_measurer = None

        if init_selected is not None and "sdata" in init_selected.metadata:
            if isinstance(
                init_selected.data,
                napari.layers._multiscale_data.MultiScaleData,
            ):
                init_model_masker = TMAMasker(
                    init_selected.metadata["sdata"],
                    init_selected.metadata["name"],
                )

                init_model_segmenter = TMASegmenter(
                    init_selected.metadata["sdata"],
                    init_selected.metadata["name"],
                )

                init_model_measurer = TMAMeasurer(
                    init_selected.metadata["sdata"],
                    init_selected.metadata["name"],
                )

            if isinstance(init_selected, napari.layers.Labels):
                init_model_dearrayer = TMADearrayer(
                    init_selected.metadata["sdata"],
                    init_selected.metadata["name"],
                )

        self.masker = TMAMaskerNapariWidget(self.viewer, init_model_masker)
        #        self.masker.max_width = 475
        self.masker.max_height = 700
        self.addTab(self.masker.native, "Masker")

        self.dearrayer = TMADearrayerNapariWidget(
            self.viewer, init_model_dearrayer
        )
        #        self.dearrayer.max_width = 475
        self.dearrayer.max_height = 400
        self.addTab(self.dearrayer.native, "Dearrayer")

        self.segmenter = TMASegmenterNapariWidget(
            self.viewer, init_model_segmenter
        )
        #        self.segmenter.max_width = 475
        self.segmenter.max_height = 500
        self.addTab(self.segmenter.native, "Segmenter")

        self.measurer = TMAMeasurerNapariWidget(
            self.viewer, init_model_measurer
        )
        #        self.measurer.max_width = 475
        self.measurer.max_height = 400
        self.addTab(self.measurer.native, "ExpressionMeasurer")

        self.utils = UtilsNapariWidget(self.viewer, init_model_masker)
        #        self.utils.max_width = 475
        self.utils.max_height = 500
        self.addTab(self.utils.native, "Utilities")

    def on_selection_change(self):
        """When layer changes, do the following actions;"""
        layer = self.viewer.layers.selection.active

        if layer is not None:
            self._show_shapes_annotations(layer)

    def _show_shapes_annotations(self, layer):
        # TODO ideally this should maybe be in the shapes metadata or anndata metadata
        ANNOTATION_LABEL = "tma_label"
        if (
            "_columns_df" in layer.metadata
            and layer.metadata["_columns_df"] is not None
            and ANNOTATION_LABEL in layer.metadata["_columns_df"]
        ):
            annot_label_vals = layer.metadata["_columns_df"][ANNOTATION_LABEL]
            layer.features[ANNOTATION_LABEL] = annot_label_vals

            layer.text = ANNOTATION_LABEL
            layer.text.visible = True
            layer.text.anchor = "upper_left"
            layer.text.size = 10

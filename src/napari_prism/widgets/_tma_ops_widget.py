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

        self.utils = UtilsNapariWidget(self.viewer, init_model_masker)
        #        self.utils.max_width = 475
        self.utils.max_height = 500
        self.addTab(self.utils.native, "Utils")

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
        self.segmenter.max_height = 700
        self.addTab(self.segmenter.native, "Segmenter")

        self.measurer = TMAMeasurerNapariWidget(
            self.viewer, init_model_measurer
        )
        #        self.measurer.max_width = 475
        self.measurer.max_height = 400
        self.addTab(self.measurer.native, "ExpressionMeasurer")

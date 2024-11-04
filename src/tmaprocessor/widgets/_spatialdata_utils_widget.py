""" Widget with useful functions for working with SpatialData loaded via
    napari-spatialdata. """

# Widget for extracting and adding indidivudal channels as layers from multiscale
# images

# Some from https://github.com/scverse/napari-spatialdata/blob/main/src/napari_spatialdata/_view.py

from typing import Iterable
from qtpy.QtWidgets import (
    QComboBox,
    QGridLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from napari.viewer import Viewer
from napari_spatialdata._model import DataModel
from napari_spatialdata._widgets import AListWidget
from napari.layers import Image, Labels, Layer, Points, Shapes
from napari_spatialdata.utils._utils import NDArrayA, _min_max_norm, get_napari_version
import packaging.version
from collections import defaultdict
from napari.utils import DirectLabelColormap
import numpy as np
from anndata import AnnData

class CustomAListWidget(AListWidget):
    def __init__(self, viewer: Viewer | None, model: DataModel, attr: str, **kwargs):
        super().__init__(viewer, model, attr, **kwargs)
    
    def _onAction(self, items: Iterable[str]) -> None:
        for item in sorted(set(items)):
            if isinstance(self.model.layer, (Image)):
                i = self.model.layer.metadata["adata"].var.index.get_loc(item)
                #self.viewer.dims.set_point(0, i)
                # Add 
                active_layer = self.viewer.layers.selection.active # assume multiscale
                sliced = [x[i, :, :] for x in active_layer.data]
                self.viewer.add_image(
                    sliced,
                    rgb=False,
                    multiscale=True,
                    name=item,
                    blending="additive"
                ) 

class MultiscaleDecomposer(QWidget):
    def __init__(self, napari_viewer: Viewer, model: DataModel | None = None) -> None:
        super().__init__()

        self._viewer = napari_viewer
        self.model = model if model else napari_viewer.window._dock_widgets["SpatialData"].widget().viewer_model._model

        self._select_layer()
        self._viewer.layers.selection.events.changed.connect(self._select_layer)

        self.setLayout(QVBoxLayout())

        var_label = QLabel("Choose var to add as layers:")
        self.custom_var_widget = CustomAListWidget(self._viewer, self.model, attr="var")
        self.layout().addWidget(var_label)
        self.layout().addWidget(self.custom_var_widget)

    def _select_layer(self) -> None:
        """Napari layers."""
        layer = self._viewer.layers.selection.active
        print(layer)
        if not hasattr(layer, "metadata") or not isinstance(layer.metadata.get("adata"), AnnData):
            print("clearing custom var widget")
            self.custom_var_widget.clear()
        else:
            print("updating custom var widget")
            if hasattr(self, "custom_var_widget"):
                self.custom_var_widget._onChange()

    def _on_layer_update(self, event=None) -> None:
        self.custom_var_widget._onChange()

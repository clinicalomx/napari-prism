from typing import Any

import napari
import pandas as pd
from anndata import AnnData
from magicgui.widgets import ComboBox
from spatialdata import SpatialData

from napari_prism.widgets._widget_utils import (
    BaseNapariWidget,
    get_layer_index_by_name,
)


class AnnDataOperatorWidget(BaseNapariWidget):
    """Parent class for widgets which operate on an existing AnnData instance.
    This class holds all subclass instances. Enforces that all subclasses
    are operating on the same AnnData instance.
    """

    _instances = []

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        adata: AnnData | None = None,
        sdata: SpatialData | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            viewer: Napari viewer.
            adata: Anndata object. Usually a memory reference to the
                `AnnDataSubsetterWidget` AnnData object.
            sdata: SpatialData object.
            *args: Passed to magicgui Container init.
            **kwargs: Passed to magicgui Container init.
        """
        super().__init__(viewer, *args, **kwargs)
        self.sdata = sdata
        self.adata = adata
        self.cell_label_column = (
            "index"  # This is the default column for cell labels
        )
        self.__class__._instances.append(self)

        if adata is not None:
            self.create_model(adata)

        self._expression_selector = None
        self.create_parameter_widgets()

    def update_model(self, adata: AnnData) -> None:
        self.adata = adata

    def create_model(self, adata: AnnData) -> None:
        self.update_model(adata)

    def update_sdata(self, sdata: SpatialData) -> None:
        self.sdata = sdata

    @classmethod
    def refresh_widgets_all_operators(cls) -> None:
        """Refreshes or resets the choices of all subclass instances.
        If called on its own, does not update model, only widgets."""
        #        print(f"Refreshing all operators")
        for instance in cls._instances:
            instance.reset_choices()

    @classmethod
    def update_model_all_operators(cls, adata: AnnData) -> None:
        """Sets the working AnnData object of all subclass instances to
        `adata`. Refreshes the choices of all subclass instances according to
        attributes available in the new `adata`.

        Args:
            adata: AnnData object to set as the working object.
        """
        for instance in cls._instances:
            instance.update_model(adata)

        AnnDataOperatorWidget.refresh_widgets_all_operators()

    @classmethod
    def create_model_all_operators(cls, adata: AnnData) -> None:
        """Creates models with the working AnnData object of all subclass
        instances which use `self.create_model`. Refreshes the choices of all
        subclass instances according to attributes available in the new `adata`.
        """
        for instance in cls._instances:
            instance.create_model(adata)

        AnnDataOperatorWidget.refresh_widgets_all_operators()

    @classmethod
    def update_sdata_all_operators(cls, sdata: SpatialData) -> None:
        """Sets the working SpatialData object of all subclass instances to
        `sdata`. Refreshes the choices of all subclass instances according to
        attributes available in the new `sdata`.
        """
        for instance in cls._instances:
            instance.update_sdata(sdata)

        AnnDataOperatorWidget.refresh_widgets_all_operators()

    def get_obsm_keys(self, widget=None) -> list[str]:
        """Returns the keys of the obsm attribute of the AnnData object. If
        `adata` is None, returns an empty list."""
        if self.adata is None:
            return []
        else:
            return list(self.adata.obsm.keys())

    def get_obsp_keys(self, widget=None) -> list[str]:
        """Returns the keys of the obsp attribute of the AnnData object. If
        `adata` is None, returns an empty list."""
        if self.adata is None:
            return []
        else:
            return list(self.adata.obsp.keys())

    def get_obs_keys(self, widget=None) -> list[str]:
        """Returns the keys of the obs attribute of the AnnData object. If
        `adata` is None, returns an empty list."""
        if self.adata is None:
            return []
        else:
            return list(self.adata.obs.keys())

    def get_categorical_obs_keys(self, widget=None) -> list[str]:
        """Returns categorical keys of the obs attribute of the AnnData object.
        If `adata` is None, returns an empty list."""
        # Can be obs keys too
        if self.adata is None:
            return []
        else:
            return [
                x
                for x in self.adata.obs.columns
                # if pd.api.types.is_categorical_dtype(self.adata.obs[x])
                if isinstance(self.adata.obs[x].dtype, pd.CategoricalDtype)
            ]

    def get_multi_categorical_obs_keys(self, widget=None) -> list[str]:
        """Returns categorical keys of the obs attribute of the AnnData object
        which have more than 2 categories. If `adata` is None, returns an empty
        list."""
        result = self.get_categorical_obs_keys(widget)
        if result != []:
            return [
                x for x in result if len(self.adata.obs[x].cat.categories) > 2
            ]
        else:
            return result

    def get_numerical_obs_keys(self, widget=None) -> list[str]:
        """Returns numerical keys of the obs attribute of the AnnData object.
        If `adata` is None, returns an empty list."""

        if self.adata is None:
            return []
        else:
            return [
                x
                for x in self.adata.obs.columns
                if pd.api.types.is_numeric_dtype(self.adata.obs[x])
            ]

    def adata_has_expression(self, adata: AnnData) -> bool:
        """Check if the `adata` object has expression information if it contains
        objects (.obs) and features (.var), or if it has a .X matrix."""
        n_obs, n_vars = adata.shape
        return (n_obs != 0 and n_vars != 0) or adata.X

    def get_expression_layers(self, widget=None) -> list[str]:
        """Returns the keys of the layers attribute of the contained AnnData
        object. If the object has no layer but has expression information, a
        default layer named "loaded_X" is created and returned. If `adata` is
        None, returns an empty list."""
        if self.adata is None:
            return []
        else:
            if len(self.adata.layers) == 0 and self.adata_has_expression(
                self.adata
            ):
                self.adata.layers["loaded_X"] = self.adata.X
            return list(self.adata.layers)

    def get_expression_and_obsm_keys(self, widget=None) -> list[str]:
        """Returns the keys of the layers and obsm attributes of the contained
        AnnData object. If `adata` is None, returns an empty list."""

        if self.adata is None:
            return []
        else:
            keys = self.get_expression_layers(widget) + self.get_obsm_keys(
                widget
            )
            if "spatial" in keys:
                keys.remove("spatial")
            return keys

    def get_markers(self, widget=None) -> list[str]:
        """Returns the keys of the var attribute of the contained AnnData
        object."""
        if self.adata is None:
            return []
        else:
            return list(self.adata.var_names)

    def create_parameter_widgets(self) -> None:
        """Creates the default widgets for operators, a single selection box
        for selecting an expression layer from the AnnData object."""
        self._expression_selector = ComboBox(
            name="ExpressionLayers",
            choices=self.get_expression_layers,
            label="Select an expression layer",
        )
        self._expression_selector.changed.connect(
            self.set_selected_expression_layer_as_X
        )
        self._expression_selector.scrollable = True
        self.extend([self._expression_selector])

    def get_selected_expression_layer(self) -> Any:
        """Returns the expression layer selected in the expression layer."""
        expression_layer = self._expression_selector.value
        return self.adata.layers[expression_layer]

    def set_selected_expression_layer_as_X(self) -> None:
        """Sets the expression layer selected in the expression layer as the
        .X attribute of the contained AnnData object."""
        if self.adata is not None and self.adata_has_expression(self.adata):
            self.adata.X = self.get_selected_expression_layer()

    def get_segmentation_element(self) -> Any:
        """Returns the labels element from the SpatialData object that annotates
        the cells in the contained AnnData object."""
        if self.sdata is None:
            return None
        spatialdata_attrs = self.adata.uns["spatialdata_attrs"]
        seg_element_name = spatialdata_attrs["region"]
        return self.sdata[seg_element_name]

    def get_segmentation_layer(self) -> napari.layers.Labels | None:
        """Returns the layer in the Napari viewer that is the segmentation
        element which annotates the cells in the contained AnnData object. If
        the layer is not in the viewer, returns None."""

        if self.sdata is None:
            return None

        spatialdata_attrs = self.adata.uns["spatialdata_attrs"]
        seg_element_name = spatialdata_attrs["region"]
        layer_ix = get_layer_index_by_name(self.viewer, seg_element_name)
        if layer_ix is not None:
            return self.viewer.layers[layer_ix]
        else:
            print("Segmentation layer does not exist in viewer")
            return None

    def get_sdata_widget(self) -> Any:
        """Returns the elements list widget from napari-spatialdata.
        NOTE: This is accessing private APIs, but may be publicly accessible
        in the future.

        https://github.com/scverse/napari-spatialdata/issues/313
        """
        return self.viewer.window._dock_widgets["SpatialData"].widget()

    def get_sdata_view_widget(self) -> Any:
        """Returns the View (napari-spatialdata) widget.
        NOTE: This is accessing private APIs, but may be publicly accessible
        in the future.

        https://github.com/scverse/napari-spatialdata/issues/313
        """
        return self.viewer.window._dock_widgets[
            "View (napari-spatialdata)"
        ].widget()

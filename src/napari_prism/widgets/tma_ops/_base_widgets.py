from abc import abstractmethod

import napari
from magicgui.widgets import ComboBox, Container
from magicgui.widgets._concrete import FloatSpinBox

from napari_prism.models.tma_ops._tma_image import (
    MultiScaleImageOperations,
    SdataImageOperations,
    SingleScaleImageOperations,
)
from napari_prism.widgets._widget_utils import BaseNapariWidget, get_ndim_index

NS_VIEW = "View (napari-spatialdata)"


class SdataImageNapariWidget(BaseNapariWidget):
    """ViewModel for SdataImageOperations."""

    _instances = []

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: SdataImageOperations | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(viewer, *args, **kwargs)
        self.model = model
        self.__class__._instances.append(self)

        # Refresh choices when layers inserted and removed
        self.viewer.layers.events.inserted.connect(
            self.refresh_widgets_all_operators
        )
        self.viewer.layers.events.removed.connect(
            self.refresh_widgets_all_operators
        )
        # Refresh choicies when layers are edited
        self.viewer.layers.events.changed.connect(
            self.refresh_widgets_all_operators
        )

    def refresh_sdata_widget(self):
        """Temporary solution until public API calls to napari-spatialdata
        classes are released.

        The result is a refresh of the elements from the possibly updated
        SpatialData disk object in the current coordinate system of the
        active (recently processed/used) layer.

        Follow: https://github.com/scverse/napari-spatialdata/issues/312
        """
        spatial_data_widget = self.get_sdata_widget()
        selected = self.viewer.layers.selection.active
        spatial_data_widget.elements_widget._onItemChange(
            selected.metadata["_current_cs"]
        )

    def get_sdata_widget(self):
        return self.viewer.window._dock_widgets["SpatialData"].widget()

    @classmethod
    def refresh_widgets_all_operators(cls):
        for instance in cls._instances:
            instance.reset_choices()

    @abstractmethod
    def update_model(self):
        raise NotImplementedError("Calling abstract method")


class SingleScaleImageNapariWidget(SdataImageNapariWidget):
    """ViewModel for SingleScaleImageOperations."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: SingleScaleImageOperations | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(viewer, model, *args, **kwargs)

    def get_singlescale_image_layers(self, widget=None):
        # overrides
        return [
            x.name
            for x in self.viewer.layers
            if (
                isinstance(x, napari.layers.image.image.Image)
                and len(x.data.shape) == 2
            )
            or (
                isinstance(x, napari.layers.labels.labels.Labels)
                and len(x.data.shape) == 2
            )
        ]

    def create_parameter_widgets(self):
        self._image_layer_selection_widget = ComboBox(
            name="SinglescaleScales",
            choices=self.get_singlescale_image_layers,
            label="Selected layer",
        )
        self._image_layer_selection_widget.changed.connect(self.update_model)

        self.extend(
            [
                self._image_layer_selection_widget,
            ]
        )


class MultiScaleImageNapariWidget(SdataImageNapariWidget):
    """ViewModel for MultiScaleImageOperations."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: MultiScaleImageOperations | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(viewer, model, *args, **kwargs)

    def get_multiscale_image_layers(self, widget=None):
        return [
            x.name
            for x in self.viewer.layers
            if isinstance(
                x.data, napari.layers._multiscale_data.MultiScaleData
            )
        ]

    def get_multiscale_image_scales(self, widget=None) -> list[str | None]:
        if self.model is not None:
            return self.model.get_image_scales()
        else:
            return [None]

    def get_multiscale_image_shapes(self, widget=None) -> list[str | None]:
        if self.model is not None:
            return self.model.get_image_shapes()[
                ::-1
            ]  # Reverse order for GUI --> But reverse index as wel..
        else:
            return [None]

    def set_scale_index(self):
        # Keep track of reverse order; this is the real order in the sdata obj
        self.scale_index = self.get_multiscale_image_shapes()[::-1].index(
            self._image_shape_selection_widget.value
        )  # Enums values start from 1, need to -1 for zeroindexing

    def get_selected_scale(self):
        return self.get_multiscale_image_scales()[self.scale_index]

    def get_selected_channel(self):
        """The selected channel is the chosen ndim index of the viewer."""
        channel_ix = get_ndim_index(self.viewer)
        channel_val = self.get_image_channels()[channel_ix]
        return channel_val

    def get_selected_image(self):
        scale = self.get_multiscale_image_scales()[self.scale_index]
        channel_val = self.get_selected_channel()
        return self.model.get_image_by_scale(scale).sel(c=channel_val)

    def get_channels(self, widget=None):
        if self.model is not None:
            return self.model.get_image_channels()
        else:
            return [None]

    def get_channel(self) -> str:
        channels = self.model.sdata[self.model.image_name]["scale0"].coords[
            "c"
        ]
        selected_channel = channels[self.viewer.dims.current_step[0]].item()
        return selected_channel

    def get_channels_api(self) -> list[str]:
        """NOTE: to be deprecated"""
        var_widget = (
            self.viewer.window._dock_widgets[NS_VIEW].widget().var_widget
        )
        selected_var_items = [x.text() for x in var_widget.selectedItems()]
        return selected_var_items

    def create_parameter_widgets(self):
        self._image_shape_selection_widget = ComboBox(
            name="MultiscaleScales",
            choices=self.get_multiscale_image_shapes,
            label="Select image resolution",
        )
        self.set_scale_index()  # Set the scale index to the lowest resolution shape
        self._image_shape_selection_widget.changed.connect(
            self.set_scale_index
        )  # set on change
        self._image_shape_selection_widget.changed.connect(self.update_model)

        self.extend(
            [
                self._image_shape_selection_widget,
            ]
        )


class RangeEditInt(Container[FloatSpinBox]):
    def __init__(
        self,
        start: int = 10,
        stop: int = 30,
        step: int = 5,
        **kwargs,
    ) -> None:
        MIN_START = 1
        self.start = FloatSpinBox(
            value=start, min=MIN_START, max=999, step=step, name="start"
        )
        self.start.changed.connect(self.update_stop)

        self.stop = FloatSpinBox(
            value=stop, min=MIN_START + step, max=999, step=step, name="stop"
        )

        self.step = FloatSpinBox(
            value=step, min=step, max=999, step=step, name="step"
        )

        kwargs["widgets"] = [self.start, self.stop, self.step]
        kwargs.setdefault("layout", "horizontal")
        kwargs.setdefault("labels", True)
        kwargs.pop("nullable", None)  # type: ignore [typeddict-item]
        super().__init__(**kwargs)

    def update_stop(self):
        """Update stop value based on the value of start and step.
        Ensure that the stop value is always greater than or equal to
        the start value, by step. Update minimum value of stop to always be
        greater than  start by step.
        """
        if self.stop.value < self.start.value:
            self.stop.value = self.start.value
        self.stop.min = self.start.value

    @property
    def value(self) -> tuple:
        """Return current value of the widget. Contrary to native,
        return tuple."""
        # modify values
        return self.start.value, self.stop.value, self.step.value

    @value.setter
    def value(self, value: tuple[float, float, float]) -> None:
        self.start.value, self.stop.value, self.step.value = value

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<{self.__class__.__name__} value={self.value!r}>"


class RangeEditFloat(Container[FloatSpinBox]):
    """Version of native RangeEdit which allows for floats. -> downstream Only
    compatible with np.arange, not native python range.

    Tuples instead, due to base python `range` not supporting floats.

    Enforces a gap between start and stop, based on step.
    """

    def __init__(
        self,
        start: float = 0.05,
        stop: float = 1.0,
        step: float = 0.01,
        **kwargs,
    ) -> None:
        MIN_START = 0.01
        self.start = FloatSpinBox(
            value=start, min=MIN_START, max=999, step=step, name="start"
        )
        self.start.changed.connect(self.update_stop)

        self.stop = FloatSpinBox(
            value=stop, min=MIN_START + step, max=999, step=step, name="stop"
        )

        self.step = FloatSpinBox(
            value=step, min=step, max=999, step=step, name="step"
        )

        kwargs["widgets"] = [self.start, self.stop, self.step]
        kwargs.setdefault("layout", "horizontal")
        kwargs.setdefault("labels", True)
        kwargs.pop("nullable", None)  # type: ignore [typeddict-item]
        super().__init__(**kwargs)

    def update_stop(self):
        """Update stop value based on the value of start and step.
        Ensure that the stop value is always greater than the start value,
        by step. Update minimum value of stop to always be greater than
        start by step.
        """
        if self.stop.value < self.start.value:
            self.stop.value = self.start.value
        self.stop.min = self.start.value

    @property
    def value(self) -> tuple:
        """Return current value of the widget. Contrary to native,
        return tuple."""
        # modify values
        return self.start.value, self.stop.value, self.step.value

    @value.setter
    def value(self, value: tuple[float, float, float]) -> None:
        self.start.value, self.stop.value, self.step.value = value

    def __repr__(self) -> str:
        """Return string representation."""
        return f"<{self.__class__.__name__} value={self.value!r}>"

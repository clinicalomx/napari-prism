from abc import abstractmethod
from enum import Enum

import napari
from magicgui.widgets import Container, Table, create_widget
from magicgui.widgets._concrete import FloatSpinBox
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QTableWidgetItem

NS_VIEW = "View (napari-spatialdata)"


def make_unique_sdata_element_name(sdata, element_name):
    count = 1
    while element_name in sdata:
        element_name += str(count)
        count += 1
    return element_name


def get_ndim_index(viewer):
    return viewer.dims.current_step[0]


def get_layer_names(viewer):
    return [x.name for x in viewer.layers]


def get_layer_index_by_name(viewer, name):
    layer_names = get_layer_names(viewer)
    if name not in layer_names:
        return None
    else:
        return layer_names.index(name)


def gpu_available():
    try:
        import rapids_singlecell  # type: ignore # noqa: F401

        return True
    except ImportError:
        return False


def create_static_selection_widget(
    selection_name, valid_options, widget_message, default_index=0
):
    """Creates a layer selection widget based on valid options. Choices
    immutable after creation."""
    Opts = Enum(selection_name, valid_options)

    # Sentinel value, if nullable, then pass value as None
    if default_index is None:
        default_value = None

    default_value = list(Opts)[default_index]

    return create_widget(
        value=default_value, name=widget_message, annotation=Opts
    )


def get_selected_layer(viewer, select_layer_widget):
    if isinstance(select_layer_widget, str):
        layer_name = select_layer_widget
    else:
        layer_name = select_layer_widget.value
    return viewer.layers[get_layer_index_by_name(viewer, layer_name)]


class BaseNapariWidget(Container):
    """MVVM paradigm. All are magicgui Containers."""

    parent_layer = None  # class attr

    def __init__(self, viewer: "napari.viewer.Viewer", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.viewer = viewer
        self.current_layer = None
        # Reset widget choices when new things are added to the viewer
        self.viewer.layers.events.inserted.connect(self.reset_choices)

    @abstractmethod
    def update_model(self):
        raise NotImplementedError("Calling abstract method")

    @abstractmethod
    def create_parameter_widgets(self):
        raise NotImplementedError("Calling abstract method")

    @classmethod
    def set_parent_layer(cls, layer):
        """Have all subclasses point to the same parent layer.
        Temporary solution until stable implementations of napari layer
        groups are available.
        """
        BaseNapariWidget.parent_layer = layer

    @classmethod
    def get_parent_layer(cls):
        return BaseNapariWidget.parent_layer


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


class EditableTable(Table):
    """Hack by making headers the first row, then hiding headers."""

    def __init__(self, value, *args, **kwargs):
        value = self.drop_key_to_val(value)
        super().__init__(*args, value=value, **kwargs)
        self.init_ui()

    @staticmethod
    def drop_key_to_val(value):
        new_d = {}
        for i, item in enumerate(value.items()):
            k, v = item
            new_list = [k] + v  # [key, labels, ...]
            new_d[i] = new_list

        return new_d

    @staticmethod
    def reverse_drop_key_to_val(value):
        original = [x[0] for x in value]  # first col
        new = [x[1] for x in value]  # rest
        reverse_d = {}
        for v in [original, new]:
            reverse_d[v[0]] = v[1:]

        return reverse_d

    def init_ui(self):
        # Make second one editable
        # self.table.horizontalHeader().sectionDoubleClicked.connect(self.changeHorizontalHeader)

        # Hide the vertical header
        self.native.verticalHeader().setVisible(False)

        # Hide the horizontal header
        self.native.horizontalHeader().setVisible(False)

        # Make the first row header like; align centre
        for col in range(self.native.columnCount()):
            frow_item = self.native.item(0, col)
            frow_item.setTextAlignment(Qt.AlignHCenter)

        # Make the first column uneditable
        for row in range(self.native.rowCount()):
            item = self.native.item(row, 0)
            if item is None:
                item = QTableWidgetItem()
                self.native.setItem(row, 0, item)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

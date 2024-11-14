from typing import Any

import matplotlib
import napari
from magicgui.widgets import ComboBox, create_widget
from qtpy.QtWidgets import QVBoxLayout, QWidget

matplotlib.use("Qt5Agg")
from enum import Enum

import matplotlib.style as mplstyle
import seaborn as sns
from matplotlib.backends.backend_qtagg import (
    FigureCanvas,
    NavigationToolbar2QT,
)
from matplotlib.figure import Figure
from napari_matplotlib.base import BaseNapariMPLWidget


class GeneralMPLWidget(BaseNapariMPLWidget):
    """Base widget for widgets which show matplotlib-compatible plots
    (i.e. matplotlib, seaborn, scanpy plots). Handles automatic figure and axes
    creation, layout clearing, with default 'tight' layout, and uses the current
    theme of the napari viewer."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        """
        Args:
            viewer: The napari viewer.
            parent: The parent widget this widget is attached to.
        """
        super().__init__(viewer, parent)
        self.add_single_axes()
        self.canvas.figure.set_layout_engine("tight")

    def plot(self, *args, **kwargs) -> None:
        """Plot the data with the current theme. Clears any previous plots.

        Args:
            *args: Passed to `self._plot`
            **kwargs: Passed to `self._plot`
        """
        self.clear()
        self.add_single_axes()

        with mplstyle.context(self.napari_theme_style_sheet):
            self._plot(*args, **kwargs)

        self.canvas.draw_idle()

    def clear(self) -> None:
        """Clear the current contained figure and axes."""
        self.figure.clear()
        self.axes.clear()

    def _plot(self):
        """Abstract method to be implemented by subclasses. This is where the
        actual plotting should be done."""
        print("Abstract method, must be implmented.")


class HistogramPlotCanvas(GeneralMPLWidget):
    """Widget for plotting a histogram with lower and upper bound vertical
    lines."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)

        #: The lower bound vertical line.
        self.lower_vline = None

        #: The lower bound vertical line annotation (value, quantile, etc).
        self.lower_vline_annot = None

        #: The upper bound vertical line.
        self.upper_vline = None

        #: The upper bound vertical line annotation (value, quantile, etc).
        self.upper_vline_annot = None

        #: The data being plotted.
        self.data = None

    def update_lines(
        self,
        lower_bound: float,
        upper_bound: float,
        lower_bound_label: str,
        upper_bound_label: str,
    ) -> None:
        """Update the vertical lines and its annotations.

        Args:
            lower_bound: The new lower bound value.
            upper_bound: The new upper bound value.
            lower_bound_label: The new lower bound label.
            upper_bound_label: The new upper bound label.
        """
        if self.lower_vline is not None:
            self.lower_vline.set_xdata([lower_bound])
            self.lower_vline_annot.set_x(lower_bound - 0.01)
            self.lower_vline_annot.set_text(lower_bound_label)

        if self.upper_vline is not None:
            self.upper_vline.set_xdata([upper_bound])
            self.upper_vline_annot.set_x(upper_bound + 0.01)
            self.upper_vline_annot.set_text(upper_bound_label)

        self.canvas.draw_idle()

    def _plot(
        self,
        data: Any,
        nbins: int,
        figsize: tuple[int, int],
        min_val: int | None = None,
        max_val: int | None = None,
        vline_min: float | None = None,
        vline_max: float | None = None,
        vline_min_label: str | None = None,
        vline_max_label: str | None = None,
    ) -> None:
        """Plots the histogram with the given data and parameters.

        Uses the following matplotlib funtions:
        - matplotlib.axes.Axes.hist to plot the histogram.
        - matplotlib.axes.Axes.axvline to plot the vertical lines.
        - matplotlib.axes.Axes.text to plot the annotations.

        Args:
            data: The data to plot.
            nbins: The number of bins to use. If 0, uses 'auto'.
            figsize: The size of the figure.
            min_val: The minimum value for the x-axis.
            max_val: The maximum value for the x-axis.
            vline_min: The value for the lower bound vertical line.
            vline_max: The value for the upper bound vertical line.
            vline_min_label: The label for the lower bound vertical line.
            vline_max_label: The label for the upper bound vertical
        """
        self.canvas.figure.set_figwidth(figsize[0])
        self.canvas.figure.set_figheight(figsize[1])

        # Cache data
        bins = "auto" if nbins == 0 else int(nbins)

        hist, plot_bins, _ = self.axes.hist(
            data, bins=bins, range=(min_val, max_val)
        )

        if vline_min is not None:
            self.lower_vline = self.axes.axvline(vline_min, color="r")
            self.lower_vline_annot = self.axes.text(
                vline_min,
                0.99,
                vline_min_label,
                ha="right",
                va="top",
                transform=self.axes.get_xaxis_transform(),
                color="r",
            )

        if vline_max is not None:
            self.upper_vline = self.axes.axvline(vline_max, color="r")
            self.upper_vline_annot = self.axes.text(
                vline_max,
                0.99,
                vline_max_label,
                ha="left",
                va="top",
                transform=self.axes.get_xaxis_transform(),
                color="r",
            )


class LinePlotCanvas(GeneralMPLWidget):
    """Widget for plotting a line plot with seaborn."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)

    def _plot(self, *args, **kwargs):
        sns.lineplot(*args, ax=self.axes, **kwargs)


class HeatmapPlotCanvas(GeneralMPLWidget):
    """Widget for plotting a heatmap with seaborn."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)

    def _plot(
        self,
        data: Any,
        cmap: str,
        vmin: float,
        vmax: float,
        vcenter: float,
        figsize: tuple[int, int] | None = None,
    ):
        """Plots the heatmap with the given data and parameters.

        Args:
            data: The data to plot.
            cmap: The colormap to use.
            vmin: The minimum value for the color scale.
            vmax: The maximum value for the color scale.
            vcenter: The center value for the color scale.
            figsize: The size of the figure.
        """
        if figsize is not None:
            self.canvas.figure.set_figwidth(figsize[0])
            self.canvas.figure.set_figheight(figsize[1])

        sns.heatmap(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            center=vcenter,
            ax=self.axes,
        )


class ClusterEvaluatorPlotCanvas(QWidget):
    """Widget for plotting the cluster evaluation scores for the
    ClusterAssessment class. TODO: implement according to above."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model  # Clusteval
        # PARAMS
        # K-R selection -> Each R operates on the same K but not the other way around.

        scores = [
            "Adjusted Rand Index",
            "Normalized Mutual Info",
            "Adjusted Mutual Info",
        ]

        Opts = Enum("ClusterScores", scores)
        iterable_opts = list(Opts)
        self.score_list = create_widget(
            value=iterable_opts[0],  # standard
            name="Cluster Scores",
            widget_type="ComboBox",
            annotation=Opts,
        )
        self.score_list.scrollable = True
        self.score_list.changed.connect(self.update_plot)

        self.ks_selection = ComboBox(
            name="KParam",
            choices=self.get_ks,
            label="Subset by K parameter",
            nullable=True,
        )
        self.ks_selection.scrollable = True
        self.ks_selection.changed.connect(self.update_plot)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.score_list.native)
        self.layout.addWidget(self.ks_selection.native)

        self.fig = None
        self.ax = None
        self.canvas = FigureCanvas()  # FigureCanvasQTAgg()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)

        # K, R selection

        self.update_plot()

    def get_ks(self, widget=None) -> None:
        """Get the available K values from the model."""
        if self.model is None:
            return []
        else:
            return self.model.adata.uns["param_grid"]["ks"]

    def update_plot(self) -> None:
        """Update the plot with the current model and parameters."""
        if self.fig is not None:
            self.fig.clear()
        if self.ax is not None:
            self.ax.clear()
        self.fig = Figure(figsize=(5, 5))
        self.canvas.figure = self.fig
        self.ax = self.fig.add_subplot(111)

        if self.model is not None:
            score_getters = {
                "Adjusted Rand Index": self.model.adjusted_rand_index,
                "Normalized Mutual Info": self.model.normalized_mutual_info,
                "Adjusted Mutual Info": self.model.adjusted_mutual_info,
            }
            k = self.ks_selection.value
            if k is not None:
                k = int(k)
            score_df = score_getters[self.score_list.value.name](k)
            sns.heatmap(
                score_df, ax=self.ax, cmap="viridis", vmin=0, vmax=1
            )  # Is none, empty plot.
            self.ax.figure.canvas.mpl_connect("pick_event", lambda x: print(x))
        self.canvas.draw_idle()

    def set_new_model(self, model: Any) -> None:
        """Set a new model to use for plotting."""
        self.model = model
        self.update_plot()


# NOTE feature modelling type plot
class ComplexHeatmapPlotCanvas(GeneralMPLWidget):
    """Widget for plotting PyComplexHeatmaps"""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)

    def _plot(self):
        raise NotImplementedError()

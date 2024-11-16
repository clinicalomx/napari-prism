import decimal
from enum import Enum

import loguru
import napari
import numpy as np
from anndata import AnnData
from magicgui.widgets import ComboBox, Container, Select, Table, create_widget
from napari.qt.threading import thread_worker
from napari.utils.events import EmitterGroup
from qtpy.QtCore import QPoint, Qt, QTimer
from qtpy.QtWidgets import QAction, QMenu, QTabWidget, QTreeWidget
from superqt import QLabeledDoubleRangeSlider, QLabeledSlider
from superqt.sliders import MONTEREY_SLIDER_STYLES_FIX

from napari_prism import pp  # refactored preprocessing class to funcs only
from napari_prism.models.adata_ops.cell_typing._augmentation import (
    add_obs_as_var,
    subset_adata_by_var,
)
from napari_prism.models.adata_ops.cell_typing._clusteval import (
    ClusteringSearchEvaluator,
)
from napari_prism.models.adata_ops.cell_typing._clustsearch import (
    HybridPhenographSearch,
    ScanpyClusteringSearch,
)
from napari_prism.models.adata_ops.cell_typing._subsetter import AnnDataNodeQT
from napari_prism.widgets._widget_utils import (
    BaseNapariWidget,
    EditableTable,
    RangeEditFloat,
    RangeEditInt,
    gpu_available,
)
from napari_prism.widgets.adata_ops._base_widgets import AnnDataOperatorWidget
from napari_prism.widgets.adata_ops._plot_widgets import (
    ClusterEvaluatorPlotCanvas,
    HistogramPlotCanvas,
)
from napari_prism.widgets.adata_ops._scanpy_widgets import (
    ScanpyFunctionWidget,
    ScanpyPlotWidget,
)


class AnnDataSubsetterWidget(BaseNapariWidget):
    """Widget for subsetting anndata objects. Holds a QTreeWidget for
    organising AnnData objects in a tree-like structure."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        adata: AnnData | None = None,
        *args,
        **kwargs,
    ) -> None:
        """
        Args:
            viewer: Napari viewer.
            adata: Anndata object. Defaults to None in the case the user has no
                available anndata object to work with.
            *args: Passed to magicgui Container init.
            **kwargs: Passed to magicgui Container init.
        """
        super().__init__(viewer, *args, **kwargs)

        #: In-memory anndata object.
        self.adata = adata

        #: Events for when an anndata object is created, changed, or saved.
        self.events = EmitterGroup(
            source=self,
            adata_created=None,
            adata_changed=None,
            adata_saved=None,
        )

        #: Create the root node for the tree widget.
        if adata is not None:
            self.create_model(adata)

        #: Create the widgets.
        self.adata_tree_widget = None
        self.create_parameter_widgets()

    def create_model(self, adata: AnnData, emit: bool = True) -> None:
        """Creates an entirely new Tree, usually by changing
        the image parent.

        Args:
            adata: Anndata object.
            emit: Whether to emit the `self.adata_created` event. Defaults to
                True.
        """
        self.adata = adata
        layout = self.native.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

        if emit:
            self.events.adata_created(value=self.adata)

        self.create_parameter_widgets()

    def update_model(self, adata: AnnData, emit: bool = True) -> None:
        """Update the selected AnnData node in the tree widget. Broadcasts this
        new AnnData object to all listeners.

        Args:
            adata: Newly selected Anndata object in the tree widget.
            emit: Whether to emit the `self.adata_changed` event. Defaults to
                True.
        """
        self.adata = adata
        if emit:
            self.events.adata_changed(value=self.adata)

    def refresh_node_labels(self) -> None:
        """Updates the labels, info, tooltips of the AnnData nodes in the tree.

        When the AnnData object is changed or new attributes are added, these
        aren't automatically updated in the tree widget, since the labels are
        set at the time of creation.

        TODO
        """
        raise NotImplementedError("Not yet implemented")

    def add_anndata_node(self, adata: AnnData) -> None:
        """Add an anndata node to the currently selected node in the tree
        widget.

        Args:
            adata: Anndata object.
        """
        adata_node = AnnDataNodeQT(adata, None, "Root", self.adata_tree_widget)
        for column in range(self.adata_tree_widget.columnCount()):
            self.adata_tree_widget.resizeColumnToContents(column)

        self.adata_tree_widget.setCurrentItem(adata_node)

        self.adata_tree_widget.currentItemChanged.connect(
            lambda x: self.update_model(x.adata)
        )

    def show_context_menu(self, pos: QPoint) -> None:
        """Show the context menu at the user cursor when right-clicking on a
        node in the tree.

        Context menu options:
            - Save: Save the current node to the viewer.
            - Annotate Obs: Launch the table annotation widget. TODO
            - Delete: Delete the current node. Option only available if the
                node is not the root node.

        Args:
            pos: QPoint of the current position of the user's cursor
        """
        item = self.adata_tree_widget.itemAt(pos)

        if item:
            context_menu = QMenu()
            # save action
            save_action = QAction("Save", self.native)
            save_action.triggered.connect(lambda: self.save_current_node())
            context_menu.addAction(save_action)

            # annotate action; TODO
            annotate_action = QAction("Annotate Obs", self.native)
            annotate_action.triggered.connect(lambda: self.annotate_node_obs())
            context_menu.addAction(annotate_action)

            # delete action. Root node cannot be deleted.
            if item.text(0) != "Root":
                delete_action = QAction("Delete", self.native)
                delete_action.triggered.connect(lambda: self.delete_node(item))
                context_menu.addAction(delete_action)

            context_menu.exec_(self.adata_tree_widget.mapToGlobal(pos))

    def save_current_node(self) -> None:
        """Call to save the current AnnData node directly to the underlying
        on-disk SpatialData object. Handled by the `AnnDataAnalsisParentWidget`.
        """
        current_node = self.adata_tree_widget.currentItem()
        current_node.inherit_children_obs()
        adata_out = current_node.adata
        self.events.adata_saved(value=adata_out)

    def delete_node(self, node: AnnDataNodeQT) -> None:
        """Deletes the current AnnData node and its children from the tree
        widget.
        """
        parent = node.parent()
        if parent:
            parent.removeChild(node)
        else:
            self.adata_tree_widget.takeTopLevelItem(
                self.adata_tree_widget.indexOfTopLevelItem(node)
            )

    def annotate_node_obs(self) -> None:
        """Launch the table annotation widget for the current node.

        This widget allows the user to annotate the obs columns of the current
        node in an excel-like table entry, or by importing a CSV file.
        """
        # current_node = self.adata_tree_widget.currentItem()
        raise NotImplementedError("Not yet implemented")

    def launch_table_annotation_widget(self) -> None:
        raise NotImplementedError("Not yet implemented")

    def create_parameter_widgets(self) -> None:
        """Create the AnnData tree widget. Adds the root node to the tree if
        an anndata object is available.
        """
        self.adata_tree_widget = QTreeWidget()
        self.adata_tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.adata_tree_widget.customContextMenuRequested.connect(
            self.show_context_menu
        )
        self.native.layout().addWidget(self.adata_tree_widget)

        HEADERS = ("AnnData Subset", "Properties")
        self.adata_tree_widget.setColumnCount(len(HEADERS))
        self.adata_tree_widget.setHeaderLabels(HEADERS)

        if self.adata is not None:
            self.add_anndata_node(self.adata)

    def add_node_to_current(self, adata_slice, node_label, obs_labels=None):
        """Add a new node to the currently selected node in the tree widget. If
        the new node label already exists, it will not be added.
        """
        matches = self.adata_tree_widget.findItems(
            node_label, Qt.MatchRecursive, 0
        )

        if matches == []:
            # TODO: allow overwrite ?
            AnnDataNodeQT(
                adata_slice,
                obs_labels,
                node_label,
                parent=self.adata_tree_widget.currentItem(),
            )


class AugmentationWidget(AnnDataOperatorWidget):
    """Widget for augmenting (adding vars, subsetting by vars) AnnData objects."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an anndata object is augmented (created)
        self.events = EmitterGroup(source=self, augment_created=None)
        super().__init__(viewer, adata)

    def reset_choices(self) -> None:
        """Reset the choices of the widgets in the widget. Propagate this to
        the children widgets.
        """
        super().reset_choices()
        self.additive_aug.reset_choices()
        self.reductive_aug.reset_choices()

    def create_parameter_widgets(self) -> None:
        """Creates two tabs for additive and reductive augmentation."""
        self.augmentation_tabs = QTabWidget()
        self.native.layout().addWidget(self.augmentation_tabs)

        self._expression_selector = ComboBox(
            name="ExpressionLayers",
            choices=self.get_expression_layers,
            label="Select an expression layer",
        )
        self._expression_selector.scrollable = True

        self.obs_selection = Select(
            name="ObsKeys",
            choices=self.get_numerical_obs_keys,  # numerical only, string breaks
            label="Select obs keys to add as features",
            value=None,
            nullable=True,
        )
        self.obs_selection.scrollable = True
        self.obs_selection.changed.connect(self.reset_choices)

        self.add_obs_as_var_button = create_widget(
            name="Add as feature", widget_type="PushButton", annotation=bool
        )
        self.add_obs_as_var_button.changed.connect(self._add_obs_as_var)

        self.var_selection = Select(
            name="VarKeys",
            choices=self.get_markers,
            label="Select var keys to subset by",
            value=None,
            nullable=True,
        )
        self.var_selection.scrollable = True

        self.subset_var_button = create_widget(
            name="Subset by var", widget_type="PushButton", annotation=bool
        )
        self.subset_var_button.changed.connect(self._subset_by_var)

        self.additive_aug = Container()
        self.additive_aug.extend(
            [
                self._expression_selector,
                self.obs_selection,
                self.add_obs_as_var_button,
            ]
        )

        self.reductive_aug = Container()
        self.reductive_aug.extend([self.var_selection, self.subset_var_button])

        self.augmentation_tabs.addTab(
            self.additive_aug.native, "Additive Augmentation"
        )

        self.augmentation_tabs.addTab(
            self.reductive_aug.native, "Reductive Augmentation"
        )

    def get_markers(self, widget=None) -> list[str]:
        """Get the .var keys from the AnnData object."""
        if self.adata is None:
            return []
        else:
            return list(self.adata.var_names)

    def _subset_by_var(self) -> None:
        """Create a View of an AnnData subset by the select var key(s). Add this
        as a new child node labelled by the var keys separated by an underscore
        to the original parent AnnData.
        """
        var_keys = self.var_selection.value
        if var_keys != []:
            aug_adata = subset_adata_by_var(self.adata, var_keys)
            node_label = "subset" + "_".join(var_keys)
            self.events.augment_created(value=(aug_adata, node_label))

    def _add_obs_as_var(self) -> None:
        """Create a new AnnData object with the selected obs keys added as
        features. Add this as a new child node labelled by the obs keys
        separated by an underscore to the original parent AnnData.
        """
        obs_keys = self.obs_selection.value
        layer_key = self._expression_selector.value
        node_label = "" if layer_key is None else layer_key
        if obs_keys[0] is not None:
            aug_adata = add_obs_as_var(self.adata, obs_keys, layer_key)
            node_label += f"_{'_'.join(obs_keys)}"
            self.events.augment_created(value=(aug_adata, node_label))

    def get_obs_keys(self, widget=None) -> list[str]:
        """Get the .obs keys from the AnnData object."""
        if self.adata is None:
            return []
        else:
            return list(self.adata.obs.columns)


class QCWidget(AnnDataOperatorWidget):
    """Widget for quality control and filtering of AnnData objects."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an anndata object is augmented (created)
        self.events = EmitterGroup(source=self, augment_created=None)
        super().__init__(viewer, adata)

        #: Range slider for the upper and lower bound of histogram plots
        self.range_slider = None

        #: Slider for the number of bins in the histogram plots
        self.nbins_slider = None

        #: Variable selection widgets
        self.obs_selection = None
        self.var_selection = None

        #: Canvas placeholder for the histogram plots
        self.hist_canvas = None

        #: Directive for subsetting, either by value or quantile
        self.current_value_directive = None

        #: Current key or attribute in the AnnData object to filter and qc by
        self.current_key = "obs"

        #: Current expression layer in the AnnData object to filter and qc by
        self.current_layer = None

    def update_layer(self, layer: str) -> None:
        self.current_layer = layer

    def create_parameter_widgets(self) -> None:
        """Dynamically create parameter widgets for the QC widget. Starts off as
        a single ComboBox. Once a QC function is selected, the widget will call
        `self.local_create_parameter_widgets` to create the appropriate widgets.
        """
        self.qc_functions = {
            "filter_by_obs_count": pp.filter_by_obs_count,
            "filter_by_obs_value": pp.filter_by_obs_value,
            "filter_by_obs_quantile": pp.filter_by_obs_quantile,
            "filter_by_var_value": pp.filter_by_var_value,
            "filter_by_var_quantile": pp.filter_by_var_quantile,
        }
        Opts = Enum("QCFunctions", list(self.qc_functions.keys()))

        self.qc_selection = create_widget(
            value=None,
            name="QC function",
            widget_type="ComboBox",
            annotation=Opts,
            options={"nullable": True},
        )
        self.qc_selection.scrollable = True
        self.qc_selection.changed.connect(self.local_create_parameter_widgets)

        self.extend([self.qc_selection])

    def clear_local_layout(self) -> None:
        """Clear the layout of the locally create widgets. Keeps the original
        QC selection widget, and the apply button."""
        layout = self.native.layout()
        # dont remove the first
        # Remove first item continually until the last
        for _ in range(layout.count() - 1):
            layout.itemAt(1).widget().setParent(None)

        if self.hist_canvas is not None:
            self.hist_canvas.clear()

    def create_range_sliders(self) -> None:
        """Create the range sliders for the histogram plots. Updates the plot
        with vertical lines corresponding to the value of the sliders."""
        self.range_slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self.range_slider.setHandleLabelPosition(
            QLabeledDoubleRangeSlider.LabelPosition.NoLabel
        )

        self.range_slider.setStyleSheet(
            MONTEREY_SLIDER_STYLES_FIX
        )  # macos fix
        self.range_slider.valueChanged.connect(self.update_lines)
        self.native.layout().addWidget(self.range_slider)

    def create_histogram_plot(self, value_directive="value") -> None:
        """Create the histogram plot canvas for the selected obs or var key."""
        self.hist_canvas = HistogramPlotCanvas(self.viewer, self.native)
        self.native.layout().addWidget(self.hist_canvas)

    def create_nbins_slider(self) -> None:
        """Create the slider for the number of bins in the histogram plot.
        Sets a buffer timer to prevent the plot from updating too frequently,
        as this can appear laggy to the user especially for larger nbins values.
        """
        self.nbins_slider = QLabeledSlider(Qt.Horizontal)
        self.nbins_slider.setRange(0, 500)
        self.nbins_slider.setValue(0)
        self.nbins_slider.setStyleSheet(
            MONTEREY_SLIDER_STYLES_FIX  # macos fix
        )
        self.native.layout().addWidget(self.nbins_slider)
        self.nbins_slider.valueChanged.connect(self.on_slider_moved)
        # Buffer the nbins slider so plot isnt updated too frequently
        self.update_timer = QTimer()
        self.update_timer.setInterval(200)  # Delay in milliseconds
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)

    def on_slider_moved(self) -> None:
        """Buffer the nbins slider so the plot isn't updated too frequently."""
        if self.update_timer.isActive():
            self.update_timer.stop()
        self.update_timer.start()

    def update_lines(self) -> None:
        """Update and show red vertical lines with value labels in the histogram
        plot to reflect the values of the range sliders."""
        if self.range_slider is not None:
            min_val, max_val = self.range_slider.value()
            min_val_label = f"{min_val:.2f}"  # q
            max_val_label = f"{max_val:.2f}"
            if self.current_value_directive == "quantile":
                if self.current_key == "obs":
                    min_val = np.quantile(
                        self.adata.obs[self.obs_selection.value], min_val
                    )

                    max_val = np.quantile(
                        self.adata.obs[self.obs_selection.value], max_val
                    )

                elif self.current_key == "var":
                    min_val = np.quantile(
                        self.adata[:, self.var_selection.value].layers[
                            self.current_layer
                        ],
                        min_val,
                    )

                    max_val = np.quantile(
                        self.adata[:, self.var_selection.value].layers[
                            self.current_layer
                        ],
                        max_val,
                    )

                min_val_label += f" ({min_val:.2f})"  # q (value)
                max_val_label += f" ({max_val:.2f})"

            self.hist_canvas.update_lines(
                min_val, max_val, min_val_label, max_val_label
            )

    def update_plot(self) -> None:
        """Update the histogram plot with the selected obs or var key. If
        .obs is categorical, then plots the value counts of cells in that
        category. Otherwise, plots the distribution of numerical .obs, and .var
        keys."""
        if self.adata is not None:
            if self.current_key == "obs":
                if self.obs_selection.value in self.get_categorical_obs_keys():
                    data = self.adata.obs[
                        self.obs_selection.value
                    ].value_counts()
                elif self.obs_selection.value in self.get_numerical_obs_keys():
                    data = self.adata.obs[self.obs_selection.value]
                else:
                    raise ValueError("Unchecked obs key")

            elif self.current_key == "var":
                adata_sub = self.adata[:, self.var_selection.value]
                if self.current_layer is not None:
                    data = adata_sub.layers[self.current_layer]
                else:
                    if adata_sub.X is not None:
                        print("No expression layer selected. Using .X")
                        data = adata_sub.X
                    else:
                        raise ValueError("Null expression matrices")
            else:
                raise ValueError("Unchecked current_key")

            min_val, max_val = (
                int(np.floor(min(data))),
                int(np.ceil(max(data))),
            )

            if self.current_value_directive == "quantile":
                self.range_slider.setRange(0, 1)
                self.range_slider.setValue((0, 1))
            else:
                self.range_slider.setRange(min_val, max_val)
                self.range_slider.setValue((min_val, max_val))

            nbins = 0  # auto
            if self.nbins_slider is not None:
                nbins = self.nbins_slider.value()

            vline_min, vline_max = self.range_slider.value()

            vline_min_label = f"{vline_min:.2f}"  # q
            vline_max_label = f"{vline_max:.2f}"  # q

            if self.current_value_directive == "quantile":
                vline_min = np.quantile(data, vline_min)
                vline_max = np.quantile(data, vline_max)

                vline_min_label += f" ({vline_min:.2f})"  # q (value)
                vline_max_label += f" ({vline_max:.2f})"  # q (value)

            self.hist_canvas.plot(
                data=data,
                nbins=nbins,
                figsize=(5, 5),
                min_val=min_val,
                max_val=max_val,
                vline_min=vline_min,
                vline_max=vline_max,
                vline_min_label=vline_min_label,
                vline_max_label=vline_max_label,
            )

    def _apply_qc(self) -> None:
        """Apply the selected QC function to the AnnData object, then emit the
        augmented AnnData object to broadcast to listener(s).
        """
        qc_func = self.qc_functions[self.qc_selection.value.name]
        node_label = f"{self.qc_selection.value.name}"
        aug_adata = None

        if self.current_key == "obs":
            obs_key = self.obs_selection.value
            min_val, max_val = self.range_slider.value()
            aug_adata = qc_func(self.adata, obs_key, min_val, max_val)
            if self.obs_selection.value is not None:
                node_label = node_label.replace(
                    "obs", self.obs_selection.value
                )

        else:
            var_key = self.var_selection.value
            min_val, max_val = self.range_slider.value()
            aug_adata = qc_func(
                self.adata, var_key, min_val, max_val, self.current_layer
            )
            if self.var_selection.value is not None:
                node_label = node_label.replace(
                    "var", self.var_selection.value
                )

        if aug_adata is not None:
            self.events.augment_created(value=(aug_adata, node_label))

    def local_create_parameter_widgets(self) -> None:
        """Create the appropriate widgets tailored for the selected QC
        function."""
        self.clear_local_layout()

        if self.qc_selection.value is not None:
            # Retrieve qc_selection
            qc_func_selection = self.qc_selection.value.name

            #
            if qc_func_selection == "filter_by_obs_count":
                self.current_value_directive = "value"
                self.current_key = "obs"
                self.obs_selection = ComboBox(
                    name="ObsKeys",
                    choices=self.get_categorical_obs_keys,
                    label="Filter cell populations by obs key",
                )
                self.obs_selection.scrollable = True
                self.obs_selection.changed.connect(self.update_plot)
                self.extend([self.obs_selection])

                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()

            elif qc_func_selection == "filter_by_obs_value":
                self.current_value_directive = "value"
                self.current_key = "obs"
                self.obs_selection = ComboBox(
                    name="ObsKeys",
                    choices=self.get_numerical_obs_keys,
                    label="Filter cells by obs values",
                )
                self.obs_selection.scrollable = True
                self.obs_selection.changed.connect(self.update_plot)
                self.extend([self.obs_selection])

                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
                self.create_nbins_slider()

            elif qc_func_selection == "filter_by_obs_quantile":
                self.current_value_directive = "quantile"
                self.current_key = "obs"
                self.obs_selection = ComboBox(
                    name="ObsKeys",
                    choices=self.get_numerical_obs_keys,
                    label="Filter cells by obs value quantiles",
                )
                self.obs_selection.scrollable = True
                self.obs_selection.changed.connect(self.update_plot)
                self.extend([self.obs_selection])

                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()

            elif qc_func_selection == "filter_by_var_value":
                self.current_value_directive = "value"
                self.current_key = "var"
                self.var_selection = ComboBox(
                    name="VarKeys",
                    choices=self.get_markers,
                    label="Filter cells by var values",
                )
                self.var_selection.scrollable = True
                self.var_selection.changed.connect(self.update_plot)
                self.extend([self.var_selection])
                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
                self.create_nbins_slider()

            elif qc_func_selection == "filter_by_var_quantile":
                self.current_value_directive = "quantile"
                self.current_key = "var"
                self.var_selection = ComboBox(
                    name="VarKeys",
                    choices=self.get_markers,
                    label="Filter cells by var value quantiles",
                )
                self.var_selection.scrollable = True
                self.var_selection.changed.connect(self.update_plot)
                self.extend([self.var_selection])
                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
                self.create_nbins_slider()

            else:
                print("Unchecked QC function")

            self.apply_button = create_widget(
                name="Apply QC function",
                widget_type="PushButton",
                annotation=bool,
            )
            self.apply_button.changed.connect(self._apply_qc)
            self.extend([self.apply_button])

        if self.obs_selection is not None or self.var_selection is not None:
            self.update_plot()


class PreprocessingWidget(AnnDataOperatorWidget):
    """Widget for preprocessing AnnData objects."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an anndata object is augmented (created)
        self.events = EmitterGroup(
            source=self,
            augment_created=None,  # Out
        )

        super().__init__(viewer, adata)

    def create_model(self, adata: AnnData) -> None:
        self.update_model(adata)

    def update_model(self, adata: AnnData) -> None:
        """Also creates the analysis 'model' class for the widget. Propagates
        the AnnData object and the created analysis model class to the
        embedding tab."""
        self.adata = adata
        self.embeddings_tab_cls.update_model(self.adata)

    def reset_choices(self):
        """Reset the choices of the widgets in the widget. Propagate this to
        the children widgets."""
        super().reset_choices()
        self.transform_tab.reset_choices()
        self.qc_tab.reset_choices()
        self.embeddings_tab_cls.reset_choices()

    def create_parameter_widgets(self):
        """Creates the tabs for the preprocessing widget."""
        super().create_parameter_widgets()

        # Processing Tabs
        self.processing_tabs = QTabWidget()
        self.native.layout().addWidget(self.processing_tabs)

        # Transform Tab
        self.transform_tab = Container()
        transforms = ["arcsinh", "scale", "percentile", "zscore", "log1p"]

        Opts = Enum("Transforms", transforms)
        iterable_opts = list(Opts)
        self.transforms_list = create_widget(
            value=[
                iterable_opts[0],
                iterable_opts[1],
                iterable_opts[-2],
            ],  # standard
            name="Transforms",
            widget_type="ListEdit",
            annotation=list[Opts],
            options={
                "tooltip": (
                    "Arcsinh with cofactor 150, Scale columns and rows to unit"
                    "variance, 95th percentile normalisation within columns"
                    "Z-score along rows"
                )
            },
        )
        self.transform_button = create_widget(
            name="Apply", widget_type="PushButton", annotation=bool
        )
        self.transform_button.changed.connect(self._apply_transforms)
        self.transform_tab.extend(
            [self.transforms_list, self.transform_button]
        )
        self.processing_tabs.addTab(self.transform_tab.native, "Transforms")

        # Data QC Tabs
        self.qc_tab = QCWidget(self.viewer, self.adata)
        # ingoing
        self.qc_tab.current_layer = self._expression_selector.value
        self._expression_selector.changed.connect(
            lambda x: self.qc_tab.update_layer(x)
        )

        # outgoing
        self.qc_tab.events.augment_created.connect(self.events.augment_created)

        self.processing_tabs.addTab(
            self.qc_tab.native, "Quality Control / Filtering"
        )

        self.embeddings_tab_cls = ScanpyFunctionWidget(self.viewer, self.adata)
        self.embeddings_tab_cls.current_layer = self._expression_selector.value
        self._expression_selector.changed.connect(
            lambda x: self.embeddings_tab_cls.update_layer(x)
        )
        self.processing_tabs.addTab(
            self.embeddings_tab_cls.native, "Embeddings"
        )

        # Conditionally create this widget based on gpu availability
        self.gpu_toggle_button = None
        if gpu_available():
            self.gpu_toggle_button = create_widget(
                value=False, name="Use GPU", annotation=bool
            )
            self.gpu_toggle_button.changed.connect(self._gpu_toggle)
            self.gpu_toggle_button.changed.connect(
                self.embeddings_tab_cls.gpu_toggle
            )
            self.extend([self.gpu_toggle_button])

    def _gpu_toggle(self) -> None:
        """Toggle between using the GPU or CPU version of the model."""
        if self.gpu_toggle_button.value is True:
            pp.set_backend("gpu")
        else:
            pp.set_backend("cpu")

    def _apply_transforms(self) -> None:
        """Apply the selected transforms to the selected expression layer in
        the AnnData object. Once complete, refresh all widgets to show the
        transformed expression layer in the AnnData object."""
        transform_map = {
            "arcsinh": pp.arcsinh,
            "scale": pp.scale,
            "percentile": pp.percentile,
            "zscore": pp.zscore,
            "log1p": pp.log1p,
        }

        self.set_selected_expression_layer_as_X()
        transform_label = ""
        for transform in self.transforms_list.value:
            transform_map[transform.name](self.adata, copy=False)
            transform_label += f"{transform.name}_"
        transform_label += self._expression_selector.value
        self.adata.layers[transform_label] = self.adata.X
        AnnDataOperatorWidget.refresh_widgets_all_operators()


class ClusterSearchWidget(AnnDataOperatorWidget):
    """Widget for performing multiple clustering runs of AnnData objects over
    a range of parameters."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        super().__init__(viewer, adata)

    def create_model(self, adata):
        self.adata = adata

    def update_model(self, adata):
        self.adata = adata
        self.reset_choices()

    def create_parameter_widgets(self) -> None:
        """Create widgets for the clustering search widget."""
        # user selects layers
        self.embedding_selector = ComboBox(
            name="EmbeddingLayers",
            choices=self.get_expression_and_obsm_keys,
            label="Select an embedding or expression layer",
        )
        self.embedding_selector.scrollable = True

        CLUSTER_METHODS = ["phenograph", "scanpy"]
        Opts = Enum("ClusterMethods", CLUSTER_METHODS)
        iterable_opts = list(Opts)
        self.cluster_method_list = create_widget(
            value=iterable_opts[0],
            name="Clustering Recipe",
            widget_type="ComboBox",
            annotation=Opts,
        )

        self.knn_range_edit = RangeEditInt(
            start=10, stop=30, step=5, name="K search range for KNN"
        )

        self.resolution_range_edit = RangeEditFloat(
            start=0.1,
            stop=1.0,
            step=0.1,
            name="Resolution search range for Leiden Clustering",
        )

        self.min_size_edit = create_widget(
            value=10,
            name="Minimum cluster size",
            annotation=int,
            widget_type="SpinBox",
            options={
                "tooltip": (
                    "If a cluster is found with less than this amount of cells,"
                    " then that cluster is labelled -1."
                )
            },
        )

        self.run_param_search_button = create_widget(
            name="Run Parameter Search",
            widget_type="PushButton",
            annotation=bool,
        )
        self.run_param_search_button.changed.connect(
            self.run_param_search_local
        )

        self.extend(
            [
                self.embedding_selector,
                self.cluster_method_list,
                self.knn_range_edit,
                self.resolution_range_edit,
                self.min_size_edit,
                self.run_param_search_button,
            ]
        )

    def get_available_backend(
        self, cluster_method: str = "phenograph"
    ) -> None:
        """Get the available backend for the clustering methods. If no GPU is
        available, then the backend is forced to be CPU.
        """
        # If no GPU, enforce all to be CPU
        if cluster_method == "phenograph":
            try:
                import cugraph  # type: ignore # noqa: F401
                import cuml  # type: ignore # noqa: F401

                return "GPU"
            except ImportError:
                return "CPU"
        elif cluster_method == "scanpy":
            try:
                import rapids_singlecell  # type: ignore # noqa: F401

                return "GPU"
            except ImportError:
                return "CPU"
        else:
            raise ValueError("Cluster method not recognised.")

    def _build_model(self) -> None:
        """Build the clustering model based on the selected clustering method,
        and the available backend."""
        selected_cluster_method = self.cluster_method_list.value.name
        backend = self.get_available_backend(selected_cluster_method)
        if selected_cluster_method == "phenograph":
            self.model = HybridPhenographSearch(
                knn=backend, clusterer=backend
            )  # Refiner left alone due to cpu only
        elif selected_cluster_method == "scanpy":
            self.model = ScanpyClusteringSearch(backend=backend)
        else:
            raise ValueError("Cluster method not recognised.")

    @thread_worker
    def _param_search_local(self):
        self.run_param_search_button.enabled = False
        self._build_model()

        # Validate knns
        if self.knn_range_edit.value[0] < 2:
            loguru.logger.warning("KNN minimum less than 2. Setting to 2.")

        kes = list(self.knn_range_edit.value)
        kes[1] = kes[1] + kes[2]
        ks = [int(x) for x in np.arange(*kes)]
        # print(ks)

        res = list(self.resolution_range_edit.value)
        res[1] = res[1] + res[2]  # Increment stop by step to include stop
        # Round Rs to same decimals as step due to rounding errors in arange
        decimals = decimal.Decimal(str(res[2]))
        est = decimals.as_tuple().exponent * -1
        rs = [np.round(x, decimals=est) for x in np.arange(*res)]
        # print(rs)

        min_size = int(self.min_size_edit.value)

        # Validate pca has been run
        try:
            self.adata = self.model.parameter_search(
                self.adata,
                embedding_name=self.embedding_selector.value,
                ks=ks,
                rs=rs,
                min_size=min_size,
            )

        # pass
        except ValueError as e:
            self.run_param_search_button.enabled = True
            raise ValueError(e) from None  # Just log but set to True

        self.run_param_search_button.enabled = True

    def run_param_search_local(self) -> None:
        """Run the parameter search for the selected clustering method on the
        selected expression layer in a separate processing thread. Once
        complete, refresh all widgets to show the clustering results in the
        .obsm and .uns attributes. Does this on the local machine.
        """
        worker = self._param_search_local()
        worker.start()
        worker.finished.connect(
            AnnDataOperatorWidget.refresh_widgets_all_operators
        )

    def _param_search_slurm(self):
        raise NotImplementedError("Not implemented yet")

    def run_param_search_slurm(self):
        """Run the parameter search for the selected clustering method on the
        selected expression layer using a SLURM scheduler. Launches a
        separate menu widget to configure the SLURM job and log the progress of
        the job. Once complete, notifies notification to the user.

        Useful for running GPU backends to avoid hogging GPU resources if
        running the plugin in an interactive SLURM job.

        TODO:finish dask_jobqueue backend, docs
        """
        raise NotImplementedError("Not implemented yet")


class ClusterAssessmentWidget(AnnDataOperatorWidget):
    """Widget for assessing the quality of clustering runs of AnnData objects
    from ClusterSearchWidget."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Cluster Evaluator model
        self.model = None
        super().__init__(viewer, adata)

    def create_model(self, adata):
        self.update_model(adata)

    def update_model(self, adata):
        self.adata = adata

    def update_model_local(self, run_selector_val: str) -> None:
        """Update the evaluator model based on the selected clustering run.

        Args:
            run_selector_val: The selected clustering run from the
                ClusterSearchWidget.
        """
        if self._cluster_run_selector.value is not None:
            self.model = ClusteringSearchEvaluator(
                self.adata, run_selector_val
            )
        else:
            self.model = None

        self.modularity_table.value = self.model.quality_scores

        self.cc_heatmap.set_new_model(self.model)

        self.kr_selection.reset_choices()

    def create_parameter_widgets(self) -> None:
        """Create widgets for the cluster assessment widget. Consists of a
        selector for the clustering run, another selector for the user to export
        a selected run to the AnnData object, and a plot to visualise the
        clustering stability of each run with every other run."""
        self._cluster_run_selector = ComboBox(
            name="ClusterRuns",
            choices=self.get_cluster_runs,
            label="Select a method with a parameter search run",
            nullable=True,
        )
        self._cluster_run_selector.scrollable = True
        self._cluster_run_selector.changed.connect(self.update_model_local)

        self.extend([self._cluster_run_selector])

        # Plotting Tabs
        self.plot_tabs = QTabWidget()
        self.native.layout().addWidget(self.plot_tabs)

        # All Cluster Runs
        self.cc_heatmap = ClusterEvaluatorPlotCanvas(self.model)
        self._cluster_run_selector.changed.connect(
            self.cc_heatmap.ks_selection.reset_choices
        )  # address in future
        self.plot_tabs.addTab(self.cc_heatmap, "Between-Cluster Score Plots")

        # tbl = {
        #         label_name: labels,
        #         self.DEFAULT_ANNOTATION_NAME: [None]
        #         * len(labels),  # Make header editable
        #     }
        self.modularity_table = Table()
        self.plot_tabs.addTab(
            self.modularity_table.native, "Modularity Scores"
        )
        # K/R selection
        self.kr_selection = Container(layout="horizontal", labels=True)
        self.k_selection = ComboBox(
            name="KParam", choices=self.get_ks, label="Select K", nullable=True
        )
        self.r_selection = ComboBox(
            name="RParam", choices=self.get_rs, label="Select R", nullable=True
        )
        self.kr_button = create_widget(
            name="Export Cluster Labels to Obs",
            widget_type="PushButton",
            annotation=bool,
        )
        self.kr_button.changed.connect(self.add_cluster_to_obs)
        self.kr_selection.extend(
            [self.k_selection, self.r_selection, self.kr_button]
        )
        self.extend([self.kr_selection])

    def add_cluster_to_obs(self) -> None:
        """Exports the cluster labels of the selected K and R to the .obs of
        the contained AnnData object. Refreshes all widgets to show the new
        cluster labels in .obs."""
        if self.k_selection.value is None or self.r_selection.value is None:
            return
        k = int(self.k_selection.value)
        r = float(self.r_selection.value)
        cluster_labels = self.model.get_K_R(k, r).astype(
            "category"
        )  # For viewing in obs

        self.adata.obs[f"{self._cluster_run_selector.value}_K{k}_R{r}"] = (
            cluster_labels
        )

        AnnDataOperatorWidget.refresh_widgets_all_operators()

    def get_ks(self, widget=None) -> list[str | int]:
        """Get the available K values from the clustering runs."""
        if self.model is None:
            return []
        else:
            return self.model.adata.uns["param_grid"]["ks"]

    def get_rs(self, widget=None) -> list[str | float]:
        """Get the available R values from the clustering runs."""
        if self.model is None:
            return []
        else:
            return self.model.adata.uns["param_grid"]["rs"]

    def get_cluster_runs(self, widget=None):
        """Get the available clustering runs from the AnnData object."""
        searchers = ClusteringSearchEvaluator.IMPLEMENTED_SEARCHERS

        available_runs = []
        if self.adata is None:
            return available_runs
        else:
            for searcher in searchers:
                if searcher + "_labels" in self.adata.obsm:
                    available_runs.append(searcher)

            return available_runs


class ClusterAnnotatorWidget(AnnDataOperatorWidget):
    """Widget for annotating cluster or categorical .obs columns in the AnnData
    object using a live editable annotation table and plots which visualise mean
    cluster expression values of each cluster group."""

    #: Default name of the new annotation made by the user in the editable table
    DEFAULT_ANNOTATION_NAME = "Annotation"

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        self.annotation_table = None
        super().__init__(viewer, adata)

    def create_model(self, adata: AnnData) -> None:
        self.update_model(adata)

    def update_model(self, adata: AnnData) -> None:
        self.adata = adata

    def create_parameter_widgets(self) -> None:
        """Create widgets for the cluster annotator widget. Launches a separate
        plot widget which wraps scanpy heatmap-like plots."""
        self.obs_widget = ScanpyPlotWidget(self.viewer, self.adata)
        self.obs_selection = self.obs_widget.obs_selection
        self.obs_selection.changed.connect(self.get_init_table)
        self.extend([self.obs_widget])

    def get_init_table(self, widget=None) -> None:
        """Initialise the editable table for the user to annotate the selected
        .obs key. Sorts the .obs key.

        The new annotation column can be renamed by double clicking the header.
        Annotations for each row (obs category) can be edited by entering a
        value in the cell, similar to an excel spreadsheet. Values are
        interpreted as strings.
        """
        if self.annotation_table is not None:
            self.remove(self.annotation_table)

        if self.obs_selection.value is not None:
            label_name = self.obs_selection.value
            labels = sorted(self.adata.obs[label_name].unique())
            tbl = {
                label_name: labels,
                self.DEFAULT_ANNOTATION_NAME: [None]
                * len(labels),  # Make header editable
            }

            self.annotation_table = EditableTable(tbl, name="Annotation Table")
            self.annotation_table.changed.connect(
                self.update_obs_mapping
            )  # Or connect to callback button
            self.extend([self.annotation_table])

    def update_obs_mapping(self) -> None:
        """Update the .obs column with values from the the new annotation column
        in the editable table. Refresh all widgets to show the new annotation
        column in the .obs of the contained AnnData object."""
        value = self.annotation_table.value["data"]
        d = EditableTable.reverse_drop_key_to_val(value)
        original_obs, original_labels = list(d.items())[0]
        new_obs, new_labels = list(d.items())[1]
        if (
            new_obs != self.DEFAULT_ANNOTATION_NAME
            and self.DEFAULT_ANNOTATION_NAME in self.adata.obs
        ):
            del self.adata.obs[self.DEFAULT_ANNOTATION_NAME]

        self.adata.obs[new_obs] = self.adata.obs[original_obs].map(
            dict(zip(original_labels, new_labels, strict=False))
        )
        self.adata.obs[new_obs] = self.adata.obs[new_obs].astype("category")
        AnnDataOperatorWidget.refresh_widgets_all_operators()

    def get_obs_categories(self, widget=None) -> list[str]:
        """Get the available categories from the selected .obs key."""
        if (
            self.obs_selection.value is not None
            and self.obs_selection.value[0] is not None
        ):
            obs = self.adata.obs[self.obs_selection.value[0]]
            return obs.unique()
        return []


class SubclusteringWidget(AnnDataOperatorWidget):
    """Widget for subclustering a subset of cells in the AnnData object based on
    a selected .obs key and category. User can also further subset the new
    AnnData subset to a select group of .var keys."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when a subcluster/subset is created
        self.events = EmitterGroup(source=self, subcluster_created=None)
        super().__init__(viewer, adata)

    def create_model(self, adata: AnnData) -> None:
        self.update_model(adata)

    def update_model(self, adata: AnnData) -> None:
        self.adata = adata

    def create_parameter_widgets(self) -> None:
        """Create widgets for the subclustering widget."""
        self.obs_selection = ComboBox(
            name="ObsKeys",
            choices=self.get_categorical_obs_keys,
            label="Select a cat. obs key to annotate",
            value=None,
            nullable=True,
        )
        self.obs_selection.scrollable = True
        self.obs_selection.changed.connect(self.reset_choices)

        self.obs_label_selection = ComboBox(
            name="ObsCategories",
            choices=self.get_obs_categories,
            label="Select a category to subcluster",
            value=None,
            nullable=True,
        )
        self.obs_label_selection.scrollable = True

        self.var_selection = Select(
            name="VarKeys",
            choices=self.get_markers,
            label="Select a marker to visualise",
            value=None,
            nullable=True,
        )
        self.var_selection.scrollable = True

        # TODO; this button doesnt work?
        self.subcluster_button = create_widget(
            name="Subcluster", widget_type="PushButton", annotation=bool
        )
        self.subcluster_button.changed.connect(self.subcluster)

        self.extend(
            [
                self.obs_selection,
                self.obs_label_selection,
                self.var_selection,
                self.subcluster_button,
            ]
        )

    def subcluster(self) -> None:
        """Subcluster the selected category in the selected .obs key. If a
        .var key(s) is selected, then further subset the subcluster to the
        selected .var key(s).

        Emits the new subcluster to any listener(s), labelled by selected obs
        category, with the selected .var(s) if any.
        """
        adata_subset = self.adata.copy()
        label = None
        var_suffix = ""
        # Query by obs
        if self.obs_selection.value is not None:
            obs = self.obs_selection.value
            label = self.obs_label_selection.value
            adata_subset = adata_subset[adata_subset.obs[obs] == label].copy()

        # Query by var
        if self.var_selection.value is not None:
            adata_subset = adata_subset[:, self.var_selection.value].copy()
            var_suffix = "_truncated_markers"

        # Add to tree
        node_label = label if label is not None else obs + var_suffix

        self.events.subcluster_created(value=(adata_subset, node_label))

    def get_obs_categories(self, widget=None) -> list[str]:
        """Get the available categories from the selected .obs key."""
        if self.obs_selection.value is not None:
            obs = self.adata.obs[self.obs_selection.value]
            return obs.unique()
        return []

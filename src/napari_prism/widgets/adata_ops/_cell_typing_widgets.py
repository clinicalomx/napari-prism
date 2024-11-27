import decimal
from enum import Enum
from pathlib import Path
from threading import Lock

import loguru
import napari
import numpy as np
import pandas as pd
from anndata import AnnData
from loguru import logger
from magicgui.widgets import ComboBox, Container, Select, Table, create_widget
from napari.qt.threading import thread_worker
from napari.utils.events import EmitterGroup
from qtpy.QtCore import QPoint, Qt, QTimer
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import (
    QAction,
    QFileDialog,
    QHBoxLayout,
    QMenu,
    QMessageBox,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QVBoxLayout,
    QWidget,
)
from spatialdata import SpatialData
from spatialdata.models import TableModel
from spatialdata.models.models import Schema_t
from superqt import QLabeledDoubleRangeSlider, QLabeledSlider
from superqt.sliders import MONTEREY_SLIDER_STYLES_FIX

from napari_prism import pp  # refactored preprocessing class to funcs only
from napari_prism.models.adata_ops.cell_typing._augmentation import (
    add_obs_as_var,
    subset_adata_by_obs_category,
    subset_adata_by_var,
)
from napari_prism.models.adata_ops.cell_typing._clusteval import (
    ClusteringSearchEvaluator,
)
from napari_prism.models.adata_ops.cell_typing._clustsearch import (
    HybridPhenographSearch,
    ScanpyClusteringSearch,
)
from napari_prism.models.adata_ops.cell_typing._tree import AnnDataNodeQT
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


class AnnDataTreeWidget(BaseNapariWidget):
    """Widget for holding and saving a tree of AnnData objects as nodes.

    Serves as the main ViewerModel class for handling SpatialData objects, and
    organises the chosen AnnData object for analysis.

    May be deprecated in the future in favor of napari-spatialdata's ViewerModel
    (or a class that extends it).

    Uses QTreeWidgets.
    """

    #: Lock writing operations to a single thread
    _write_lock = Lock()

    #: Default name of the new annotation made by the user in the editable table
    DEFAULT_ANNOTATION_NAME = "Annotation"

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
            adata: Anndata object. Defaults to None in the case the user has no
                available anndata object to work with.
            *args: Passed to magicgui Container init.
            **kwargs: Passed to magicgui Container init.
        """
        super().__init__(viewer, *args, **kwargs)

        #: In-memory anndata object.
        self.adata = adata

        #: Reference to main spatial data object.
        self.sdata = sdata

        #: Events for when an anndata object is created, changed, or saved.
        self.events = EmitterGroup(
            source=self,
            adata_created=None,
            adata_changed=None,
            node_changed=None,
        )

        #: Create the root node for the tree widget.
        if adata is not None:
            self.create_model(adata)

        if sdata is not None:
            self.set_sdata(sdata)

        #: Annotation table widget for the obs keys.
        self.annotation_table = None

        #: Create the widgets.
        self.adata_tree_widget = None
        self.create_parameter_widgets()

    def set_sdata(self, sdata: SpatialData) -> None:
        """Set the SpatialData object for the tree widget.

        Args:
            sdata: SpatialData object.
        """
        self.sdata = sdata
        if self.sdata is None and self.adata_tree_widget is not None:
            self.adata_tree_widget.clear()
            self.update_model(None, save=False)

    def create_model(
        self, adata: AnnData, table_path: Path, emit: bool = True
    ) -> None:
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
            self.events.adata_created(adata=self.adata)

        self.root_path = table_path
        self.create_parameter_widgets()

    def delete_from_disk(
        self, sdata: SpatialData, element_name: str, overwrite: bool
    ) -> None:
        if (
            element_name in sdata
            and len(sdata.locate_element(sdata[element_name])) != 0
        ):
            if overwrite:
                with self._write_lock:
                    logger.info(f"Overwriting {element_name}")
                    del sdata[element_name]
                    sdata.delete_element_from_disk(element_name)
            else:
                raise OSError(
                    f"`{element_name}` already exists. Use overwrite="
                    "True to rewrite."
                )

    def overwrite_element(
        self, sdata: SpatialData, element: Schema_t, element_name: str
    ) -> None:
        """Overwrite an element in the SpatialData object with a new element.
        If the element already exists, it will be replaced.

        Uses the second? workaround for incremental io:
            https://github.com/scverse/spatialdata/blob/main/tests/io/
            test_readwrite.py

        Args:
            sdata: The SpatialData object containing the element.
            element: The new element to add.
            element_name: The name of the element to add.
        """

        if sdata.is_backed():
            self.delete_from_disk(sdata, element_name, overwrite=True)
        sdata[element_name] = element
        sdata.write_element(element_name, overwrite=True)

    def save_node(
        self,
        node: AnnDataNodeQT,
        *args,
        **kwargs,
    ) -> None:
        """Adds a table to the contained SpatialData. `

        Args:
            table: The table to add.
            *args: Passed to TableModel.parse.
            **kwargs: Passed to TableModel.parse.
        """
        table = node.adata
        table_name = node.store.stem

        table = TableModel.parse(table, *args, **kwargs)
        self.overwrite_element(self.sdata, table, table_name)

    def update_model(self, adata: AnnData, save: bool = True) -> None:
        """Update the selected AnnData node in the tree widget. Broadcasts this
        new AnnData object to all listeners.

        Args:
            adata: Newly selected Anndata object in the tree widget.
            save: Whether to save the new Anndata object to disk. Defaults to
                True.
        """
        self.adata = adata

        # Save
        current_node = self.adata_tree_widget.currentItem()
        if current_node is not None:
            current_node.set_adata(adata)

            if save:
                self.save_node(current_node)

            # # Update the parents and their parents to inherit obs keys ?
            # all_parents = current_node.collect_parents()
            # if all_parents != []:
            #     for p in all_parents:
            #         p.inherit_children_obs()
            self.events.adata_changed(adata=self.adata)

    def refresh_node_labels(self) -> None:
        """Updates the labels, info, tooltips of the AnnData nodes in the tree.

        When the AnnData object is changed or new attributes are added, these
        aren't automatically updated in the tree widget, since the labels are
        set at the time of creation.

        TODO
        """
        raise NotImplementedError("Not yet implemented")

    def add_root_node(self, adata: AnnData, table_path: Path) -> None:
        """Add an anndata node as the root node to the tree widget.

        Args:
            adata: Anndata object.
        """
        init_attrs = "tree_attrs" not in adata.uns
        adata_node = AnnDataNodeQT(
            adata,
            None,
            "Root",
            self.adata_tree_widget,
            store=table_path,
            init_attrs=init_attrs,
        )

        for column in range(self.adata_tree_widget.columnCount()):
            self.adata_tree_widget.resizeColumnToContents(column)

        self.adata_tree_widget.setCurrentItem(adata_node)

        # Now we check if we have existing tables that used to be its children
        # from a previous run
        self.add_child_nodes_to_current(
            adata_node.adata, self.adata_tree_widget.currentItem()
        )

    def add_child_nodes_to_current(
        self, adata: AnnData, parent_node: AnnDataNodeQT
    ) -> None:
        """Add all child nodes to the currently selected node in the tree widget
        by reading tree structure from .uns["tree_attrs"].

        Args:
            adata: Anndata object.
        """
        children = list(adata.uns.get("tree_attrs", {}).get("children", []))
        if children != []:
            for child_path in children:
                # element_name = child.store.stem
                element_name = child_path.split("/")[-1]
                # assert element_name in self.sdata
                if element_name in self.sdata:
                    child_adata = self.sdata.tables[element_name]

                    init_attrs = "tree_attrs" not in child_adata.uns
                    node = AnnDataNodeQT(
                        child_adata,
                        None,
                        child_adata.uns["tree_attrs"]["name"],
                        parent=parent_node,
                        init_attrs=init_attrs,
                    )
                    if list(node.adata.uns["tree_attrs"]["children"]) != []:
                        node.make_uneditable()
                        self.add_child_nodes_to_current(node.adata, node)

                else:
                    # remove the child -> deleted from disk folder
                    # saved as ndarray, so
                    mask = adata.uns["tree_attrs"]["children"] != child_path
                    adata.uns["tree_attrs"]["children"] = np.delete(
                        adata.uns["tree_attrs"]["children"], mask
                    )
            self.adata_tree_widget.expandAll()

    def add_restored_node_to_current(
        self,
        adata_indices,
        node_label,
    ):
        """Adds a new node with all features restored, but subsetted to a given
        set of indices."""
        root_adata = self.adata_tree_widget.topLevelItem(0).adata
        if isinstance(root_adata.obs.index, pd.RangeIndex):
            root_adata.obs.index = root_adata.obs.index.astype(str)
        adata_slice = root_adata[adata_indices]

        # But inherit obs-wise from the parent
        current_node = self.adata_tree_widget.currentItem()
        parent_adata = current_node.adata
        parent_sliced = parent_adata[adata_indices]
        adata_slice.obs = parent_sliced.obs
        adata_slice.obsm = parent_sliced.obsm

        self.add_node_to_current(adata_slice, node_label)

    def add_node_to_current(
        self,
        adata_slice,
        node_label,
        obs_labels=None,
        save=True,
        # append_child_attr=True,
    ):
        """Add a new node to the currently selected node in the tree widget. If
        the new node label already exists, it will not be added.
        """
        matches = self.adata_tree_widget.findItems(
            node_label, Qt.MatchRecursive, 0
        )
        parent_node = self.adata_tree_widget.currentItem()
        if matches == []:
            # tree_attrs set in init
            node = AnnDataNodeQT(
                adata_slice,
                obs_labels,
                node_label,
                parent=parent_node,
            )

            # Update the children of the parent_node with this new child node
            parent_node.update_children()
            self.save_node(parent_node)

            if save:
                self.save_node(node)

            self.events.node_changed(
                table_to_add=str(node.store.stem), table_to_remove=None
            )

            # Lock the parent renaming system
            parent_node.make_uneditable()
            self.adata_tree_widget.expandAll()

    def show_context_menu(self, pos: QPoint) -> None:
        """Show the context menu at the user cursor when right-clicking on a
        node in the tree.

        Context menu options:
            - Annotate Obs: Launch the table annotation widget. TODO
            - Delete: Delete the current node. Option only available if the
                node is not the root node.

        Args:
            pos: QPoint of the current position of the user's cursor
        """
        item = self.adata_tree_widget.itemAt(pos)

        if item:
            context_menu = QMenu()

            # annotate action
            annotate_action = QAction("Annotate Obs", self.native)
            annotate_action.triggered.connect(lambda: self.annotate_node_obs())
            context_menu.addAction(annotate_action)

            # delete action. Root node cannot be deleted.
            if item.text(0) != "Root":
                delete_action = QAction("Delete", self.native)
                delete_action.triggered.connect(lambda: self.delete_node(item))
                context_menu.addAction(delete_action)

            context_menu.exec_(self.adata_tree_widget.mapToGlobal(pos))

    def delete_node(self, node: AnnDataNodeQT) -> None:
        """Deletes the current AnnData node and its children from the tree
        widget.

        Since tables on disk are flat, we can just delete via list, then just
        modify the attribute of the remaining parent node.

        """
        parent = node.parent()
        if parent:
            # collect node and children
            to_delete = [node] + node.collect_all_children()
            if self.sdata.is_backed():
                # Delete on-disk tables
                for n in to_delete:
                    self.delete_from_disk(
                        self.sdata, n.store.stem, overwrite=True
                    )
            # Then simply disconnect the node from the parent
            parent.removeChild(node)
            parent.update_children()

            # Make parent editable now that its a terminal node
            parent.make_editable()
            self.save_node(parent)

            self.events.node_changed(
                table_to_add=None, table_to_remove=str(node.store.stem)
            )
            del node

    def get_categorical_obs_keys(self, widget=None) -> list[str]:
        """Get the .obs keys from the AnnData object that are categorical."""
        if self.adata is None:
            return []
        else:
            return [
                x
                for x in self.adata.obs.columns
                # if pd.api.types.is_categorical_dtype(self.adata.obs[x])
                if isinstance(self.adata.obs[x].dtype, pd.CategoricalDtype)
            ]

    def annotate_node_obs(self) -> None:
        """Launch the table annotation widget for the current node.

        This widget allows the user to annotate the obs columns of the current
        node in an excel-like table entry, or by importing a CSV file.
        """
        # Create an obs selection, then create the table
        popout = QWidget()
        popout.setWindowTitle("Annotation Table")
        # popout.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout()
        popout.resize(200, 500)
        popout.setLayout(layout)

        obs_selection = ComboBox(
            name="ObsKeys",
            choices=self.get_categorical_obs_keys,
            label="Select obs keys to annotate",
            value=None,
            nullable=True,
        )
        obs_selection.scrollable = True
        layout.addWidget(obs_selection.native)

        button_layout = QHBoxLayout()
        annotate_button = QPushButton("Annotate")
        annotate_button.clicked.connect(
            lambda: _create_annotation_table(obs_selection.value)
        )
        csv_button = QPushButton("Import CSV")
        csv_button.clicked.connect(
            lambda: self.import_csv_metadata(obs_selection.value)
        )
        button_layout.addWidget(annotate_button)
        button_layout.addWidget(csv_button)
        layout.addLayout(button_layout)

        def _create_annotation_table(label_name):
            if self.annotation_table:  # reset table
                self.annotation_table.native.setParent(None)

            # labels = sorted(self.adata.obs[label_name].unique())
            labels = self.adata.obs[label_name].cat.categories.tolist()
            tbl = {
                label_name: labels,
                self.DEFAULT_ANNOTATION_NAME: [None]
                * len(labels),  # Make header editable
            }

            annotation_table = EditableTable(tbl, name="Annotation Table")
            self.annotation_table = annotation_table
            confirm_button = QPushButton("Confirm")
            confirm_button.clicked.connect(lambda: self.update_obs_mapping())

            # non-blocking pop out table
            layout.addWidget(annotation_table.native)
            layout.addWidget(confirm_button)

        cursor_position = QCursor.pos()
        popout.move(cursor_position)
        popout.show()
        self.annotation_table_popout = popout

    def update_obs_mapping(self) -> None:
        """Update the .obs column with values from the the new annotation column
        in the editable table. Refresh all widgets to show the new annotation
        column in the .obs of the contained AnnData object."""
        value = self.annotation_table.value["data"]
        d = EditableTable.reverse_drop_key_to_val(value)
        original_obs, original_labels = list(d.items())[0]
        new_obs, new_labels = list(d.items())[1]
        if new_obs.removesuffix("_new") == original_obs:
            new_obs = new_obs.removesuffix("_new")
        if (
            new_obs != self.DEFAULT_ANNOTATION_NAME
            and self.DEFAULT_ANNOTATION_NAME in self.adata.obs
        ):
            del self.adata.obs[self.DEFAULT_ANNOTATION_NAME]

        self.adata.obs[new_obs] = self.adata.obs[original_obs].map(
            dict(zip(original_labels, new_labels, strict=False))
        )
        self.adata.obs[new_obs] = self.adata.obs[new_obs].astype("category")

        self.update_model(self.adata)
        if self.annotation_table_popout:
            self.annotation_table = None
            self.annotation_table_popout.close()

    def import_csv_metadata(self, label_name: str) -> None:
        """Import multiple metadata columns from a .csv file using `label_name`
        as the index."""
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Select a .csv file",
            "",
            "CSV Files (*.csv)",
        )
        if not file_path:
            return

        try:
            csv_df = pd.read_csv(file_path)
            # Parse dtypes
            for col in csv_df.columns:
                try:
                    csv_df[col] = pd.to_numeric(csv_df[col], errors="raise")
                except ValueError:
                    csv_df[col] = csv_df[col].astype("category")

        except FileNotFoundError as e:
            loguru.logger.error(f"Error reading csv: {e}")
            return

        # Check if tma_labels is in csv_df
        if label_name not in csv_df.columns:
            message = f"Column {label_name} not found in csv"
            QMessageBox.warning(
                None,
                None,
                message,
                QMessageBox.Ok,
            )
            loguru.logger.error(message)
            return

        else:
            labels = sorted(self.adata.obs[label_name].unique())
            tbl = pd.DataFrame(
                {
                    label_name: labels,
                }
            )
            # MErge on tma_labels
            csv_df = csv_df.merge(tbl, how="left", on=label_name)

            # Now display the csv in a table widget
            annotation_table = EditableTable(
                csv_df.to_dict(orient="list"), name="CSV Metadata"
            )

            # Add the CSV table to the popout window
            self.annotation_table_popout.layout().addWidget(
                annotation_table.native
            )

            # Then add a confirm widget
            def _csv_obs_mapping():
                self.adata.obs = self.adata.obs.merge(
                    csv_df, on=label_name, how="right"
                )

                self.update_model(self.adata)
                self.annotation_table_popout.close()

            confirm_button = QPushButton("Confirm")
            confirm_button.clicked.connect(lambda: _csv_obs_mapping())
            self.annotation_table_popout.layout().addWidget(confirm_button)

    def create_parameter_widgets(self) -> None:
        """Create the AnnData tree widget. Adds the root node to the tree if
        an anndata object is available.
        """

        logger.info("Creating Tree Widget")
        self.adata_tree_widget = QTreeWidget()
        self.adata_tree_widget.itemChanged.connect(self.validate_node_name)

        def on_tree_item_changed(item: AnnDataNodeQT):
            save = False
            if item and hasattr(item, "adata"):
                if item.collect_children() != []:
                    item.inherit_children_obs()
                    save = True
                self.update_model(item.adata, save=save)

        self.adata_tree_widget.currentItemChanged.connect(on_tree_item_changed)
        self.adata_tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.adata_tree_widget.customContextMenuRequested.connect(
            self.show_context_menu
        )
        self.native.layout().addWidget(self.adata_tree_widget)

        HEADERS = ("AnnData Subset", "Properties")
        self.adata_tree_widget.setColumnCount(len(HEADERS))
        self.adata_tree_widget.setHeaderLabels(HEADERS)

        # set the initial column widths to 60/40 ratio each
        total_width = self.adata_tree_widget.width()
        self.adata_tree_widget.setColumnWidth(0, int(total_width * 0.6))
        self.adata_tree_widget.setColumnWidth(1, int(total_width * 0.4))

        if self.adata is not None:
            self.add_root_node(self.adata, self.root_path)

    def rename_node(self, node: AnnDataNodeQT, new_name: str) -> None:
        old_table_path = str(node.store.stem)
        node.rename(new_name)
        new_table_path = str(node.store.stem)
        # But also propagate changes to parent and child nodes
        parent = node.parent()
        if parent:
            parent.update_children()
            self.save_node(parent)

        children = node.collect_children()
        if children != []:
            for child in children:
                child.update_parent()
                new_child_name = new_name + "_" + child.text(0)
                child.update_name(new_child_name)
                self.save_node(child)

        self.events.node_changed(
            table_to_add=new_table_path, table_to_remove=old_table_path
        )

        self.adata_tree_widget.setCurrentItem(node)

    def validate_node_name(self, node: AnnDataNodeQT, column: int) -> None:
        """Rename node function. Validates before setting the node name when
        the user double clicks the node name."""
        if column != 0:
            return

        old_name = (
            node.data(0, Qt.UserRole)
            if node.data(0, Qt.UserRole)
            else node.text(0)
        )

        new_name = node.text(0).strip()

        if new_name == "Root":
            node.setText(0, old_name)
            return

        if not new_name:
            new_name = old_name

        parent = node.parent()
        sibling_count = (
            parent.childCount()
            if parent
            else self.adata_tree_widget.topLevelItemCount()
        )

        is_unique = True
        for i in range(sibling_count):
            sibling = (
                parent.child(i)
                if parent
                else self.adata_tree_widget.topLevelItem(i)
            )
            if sibling != node and sibling.text(0) == new_name:
                is_unique = False
                break

        if is_unique:
            node.setData(0, Qt.UserRole, new_name)
            if node.store.exists() and new_name != old_name:
                self.rename_node(node, new_name)
                # Temporarily unactivate the node to release the store
                self.adata_tree_widget.setCurrentItem(None)
                self.save_node(node)
                self.adata_tree_widget.setCurrentItem(node)

        else:
            node.setText(0, old_name)


class AugmentationWidget(AnnDataOperatorWidget):
    """Widget for augmenting (adding vars, subsetting by vars) AnnData objects."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an anndata object is augmented (created)
        # augment created events return the AnnData and the node label
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
        self.reductive_aug.extend(
            [
                self.var_selection,
                self.subset_var_button,
            ]
        )

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
            # node_label = "subset" + "_".join(var_keys)
            node_label = "subset_by_var"
            self.events.augment_created(adata=aug_adata, label=node_label)

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
            # node_label += f"_{'_'.join(obs_keys)}"
            node_label = "added_obs_as_var"
            self.events.augment_created(adata=aug_adata, label=node_label)


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

    def update_model(self, adata):
        # If AnnData changed, then reset qc tab.
        super().update_model(adata)
        self.qc_selection.value = None

    def update_layer(self, layer: str) -> None:
        self.current_layer = layer
        if self.qc_selection.value is not None and (
            self.obs_selection is not None or self.var_selection is not None
        ):
            self.update_plot()

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
            aug_adata = qc_func(
                self.adata, obs_key, min_val, max_val, copy=True
            )
            if self.obs_selection.value is not None:
                node_label = node_label.replace(
                    "obs", self.obs_selection.value
                )

        else:
            var_key = self.var_selection.value
            min_val, max_val = self.range_slider.value()
            aug_adata = qc_func(
                self.adata,
                var_key,
                min_val,
                max_val,
                self.current_layer,
                copy=True,
            )
            if self.var_selection.value is not None:
                node_label = node_label.replace(
                    "var", self.var_selection.value
                )

        if aug_adata is not None:
            self.events.augment_created(adata=aug_adata, label=node_label)

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
            augment_created=None,  # Passes on QCWidget output
            adata_changed=None,  # Passes on ScanpyFunctionWidget output
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

        # outgoing
        self.embeddings_tab_cls.events.adata_changed.connect(
            self.events.adata_changed
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
                self.embeddings_tab_cls.gpu_toggle  # tl is toggled there
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
            self.adata = transform_map[transform.name](
                self.adata, copy=True, layer=None
            )
            transform_label += f"{transform.name}_"
        transform_label += self._expression_selector.value
        self.adata.layers[transform_label] = self.adata.X
        self.events.adata_changed(adata=self.adata)


class ClusterSearchWidget(AnnDataOperatorWidget):
    """Widget for performing multiple clustering runs of AnnData objects over
    a range of parameters."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an anndata object is changed
        self.events = EmitterGroup(
            source=self,
            adata_changed=None,
        )
        super().__init__(viewer, adata)
        self.backend = "CPU"

    def create_model(self, adata):
        self.adata = adata

    def update_model(self, adata):
        self.adata = adata
        self.reset_choices()

    def _gpu_toggle(self) -> None:
        """Toggle between using the GPU or CPU version of the model."""
        if self.gpu_toggle_button.value is True:
            self.backend = "GPU"
        else:
            self.backend = "CPU"

    def create_parameter_widgets(self) -> None:
        """Create widgets for the clustering search widget."""
        self.gpu_toggle_button = None
        if gpu_available():
            self.gpu_toggle_button = create_widget(
                value=False, name="Use GPU", annotation=bool
            )
            self.gpu_toggle_button.changed.connect(self._gpu_toggle)
            self.extend([self.gpu_toggle_button])

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

    def _build_model(self) -> None:
        """Build the clustering model based on the selected clustering method,
        and the available backend."""
        selected_cluster_method = self.cluster_method_list.value.name
        if selected_cluster_method == "phenograph":
            self.model = HybridPhenographSearch(
                knn=self.backend, clusterer=self.backend
            )  # Refiner left alone due to cpu only
        elif selected_cluster_method == "scanpy":
            self.model = ScanpyClusteringSearch(backend=self.backend)
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
            adata = self.model.parameter_search(
                self.adata,
                embedding_name=self.embedding_selector.value,
                ks=ks,
                rs=rs,
                min_size=min_size,
            )
            self.run_param_search_button.enabled = True
            return adata

        # pass
        except ValueError as e:
            self.run_param_search_button.enabled = True
            raise ValueError(e) from None  # Just log but set to True

    def run_param_search_local(self) -> None:
        """Run the parameter search for the selected clustering method on the
        selected expression layer in a separate processing thread. Once
        complete, refresh all widgets to show the clustering results in the
        .obsm and .uns attributes. Does this on the local machine.
        """
        worker = self._param_search_local()
        worker.start()
        worker.returned.connect(lambda x: self.events.adata_changed(adata=x))

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
    from ClusterSearchWidget.

    Currently CPU only due to changes in cuml score funcs (only MI and ARI).
    """

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an anndata object is changed
        self.events = EmitterGroup(
            source=self,
            adata_changed=None,
        )
        #: Cluster Evaluator model
        self.model = None
        self.backend = "CPU"
        super().__init__(viewer, adata)

    # def _gpu_toggle(self) -> None:
    #     """Toggle between using the GPU or CPU version of the model."""
    #     if self.gpu_toggle_button.value is True:
    #         self.backend = "GPU"
    #     else:
    #         self.backend = "CPU"

    def create_model(self, adata):
        self.update_model(adata)

    def update_model(self, adata):
        self.adata = adata
        # if self.model is not None:
        #     self.update_model_local(self._cluster_run_selector.value)
        self.reset_choices()

    def update_model_local(self, run_selector_val: str) -> None:
        """Update the evaluator model based on the selected clustering run.

        Args:
            run_selector_val: The selected clustering run from the
                ClusterSearchWidget.
        """
        if self._cluster_run_selector.value is not None:
            # gpu = True if self.backend == "GPU" else False
            self.model = ClusteringSearchEvaluator(
                self.adata, run_selector_val, gpu=False
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
        # self.gpu_toggle_button = None
        # if gpu_available():
        #     self.gpu_toggle_button = create_widget(
        #         value=False, name="Use GPU", annotation=bool
        #     )
        #     self.gpu_toggle_button.changed.connect(self._gpu_toggle)
        #     self.extend([self.gpu_toggle_button])

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

        self.events.adata_changed(adata=self.adata)

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
    """Widget for visualising cluster or categorical .obs columns in the AnnData
    object using plots which visualise mean cluster expression values of each
    cluster group."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an anndata object is changed
        self.events = EmitterGroup(
            source=self,
            adata_changed=None,
        )
        self.obs_widget = None
        super().__init__(viewer, adata)

    def create_model(self, adata: AnnData) -> None:
        self.update_model(adata)

    def update_model(self, adata: AnnData) -> None:
        self.adata = adata
        if (
            self.adata is not None
            and self.obs_widget.obs_selection is not None
            and self.obs_widget._expression_selector is not None
        ):
            self.obs_widget.update_plot()
        else:
            self.obs_widget.scanpy_canvas.clear()

    def create_parameter_widgets(self) -> None:
        """Create widgets for the cluster annotator widget. Launches a separate
        plot widget which wraps scanpy heatmap-like plots."""
        self.obs_widget = ScanpyPlotWidget(self.viewer, self.adata)
        self.extend([self.obs_widget])

        # # self.obs_selection.changed.connect(self.get_init_table)

        # self.launch_annotation_table_button = create_widget(
        #     name="Launch Annotation Table",
        #     widget_type="PushButton",
        #     annotation=bool,
        # )
        # self.launch_annotation_table_button.changed.connect(
        #     self.launch_annotation_table
        # )
        # self.extend([self.obs_widget, self.launch_annotation_table_button])


class SubclusteringWidget(AnnDataOperatorWidget):
    """Widget for subclustering a subset of cells in the AnnData object based on
    a selected .obs key and category. User can also further subset the new
    AnnData subset to a select group of .var keys."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        #: Events for when an AnnData subcluster is created
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
            label="Select a cat. obs key to subcluster",
            value=None,
            nullable=True,
        )
        self.obs_selection.scrollable = True
        self.obs_selection.changed.connect(self.reset_choices)

        self.obs_label_selection = Select(
            name="ObsCategories",
            choices=self.get_obs_categories,
            label="Select a category to subcluster",
            value=None,
            nullable=True,
        )
        self.obs_label_selection.scrollable = True

        self.subcluster_button = create_widget(
            name="Subcluster", widget_type="PushButton", annotation=bool
        )
        self.subcluster_button.changed.connect(self.subcluster)

        self.extend(
            [
                self.obs_selection,
                self.obs_label_selection,
                self.subcluster_button,
            ]
        )

    def subcluster(self) -> None:
        """Subcluster the selected category in the selected .obs key.

        Emits the new subcluster to any listener(s), labelled by selected obs
        category.
        """
        obs_keys = self.obs_selection.value
        obs_labels = self.obs_label_selection.value
        if obs_keys is not None and obs_labels != []:
            aug_adata = subset_adata_by_obs_category(
                self.adata, obs_keys, obs_labels
            )
            node_label = f"subset_{obs_keys}_{'_'.join(obs_labels)}"
            self.events.subcluster_created(
                adata_indices=aug_adata.obs.index, label=node_label
            )

    def get_obs_categories(self, widget=None) -> list[str]:
        """Get the available categories from the selected .obs key."""
        if self.obs_selection.value is not None:
            obs = self.adata.obs[self.obs_selection.value]
            return obs.unique()
        return []

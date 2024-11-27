from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QLabel,
    QTreeWidgetItem,
)


# QTree Versions
class AnnDataNodeQT(QTreeWidgetItem):
    def __init__(
        self, adata, labels, name, parent, store=None, init_attrs=True
    ):
        """
        adata : AnnData
        labels : list of new cluster labels
        name : str, name of the cluster column
        parent : QTreeWidgetItem | QTreeWidget | None
        """
        #: Before init to check valid store when validating node names
        if store is not None:
            self.store = store  # Path?
        else:
            if isinstance(parent, AnnDataNodeQT):
                self.store = Path(str(parent.store) + "_" + name)
            else:
                self.store = None

        super(QTreeWidgetItem, self).__init__(parent)
        if name != "Root":
            self.make_editable()

        self.setText(0, name)
        self.setData(0, Qt.UserRole, self.text(0))

        self.labels = labels
        if labels is not None:
            if not isinstance(self.labels[0], str):
                self.labels = [str(x) for x in self.labels]

            if adata is not None:
                assert len(labels) == adata.shape[0]
                adata.obs[name] = labels

        # # And Label shown in widget
        # if name == "Root":
        #     name = self.store.stem
        # adata.uns["tree_attrs"]["name"] = name

        # # Empty child list
        # adata.uns["tree_attrs"]["children"] = []
        self.adata = adata.copy()
        if isinstance(self.adata.obs.index, pd.RangeIndex):
            self.adata.obs.index = self.adata.obs.index.astype(str)
        if init_attrs:
            self.adata.uns["tree_attrs"] = {}
            self.update_parent(call_set=False)
            self.update_children(call_set=False)
            self.update_name(name, call_set=True)
        else:
            self.set_adata(adata)

    def __repr__(self):
        def remove_after_n_obs_n_vars(input_string):
            if input_string is None:
                return input_string
            else:
                pattern = r"(n_obs\s+×\s+n_vars\s+=\s+\d+\s+×\s+\d+)"
                match = re.search(pattern, input_string)
                if match:
                    return input_string[: match.end()]
                return input_string

        out_repr = f"{remove_after_n_obs_n_vars(str(self.adata))}"

        return out_repr

    def make_uneditable(self):
        self.setFlags(self.flags() & ~Qt.ItemIsEditable)

    def make_editable(self):
        self.setFlags(self.flags() | Qt.ItemIsEditable)

    def rename(self, new_name):
        old_store = self.store
        # Force update
        self.update_name(new_name, call_set=True)

        new_table_name = self.parent().store.stem + "_" + new_name
        new_store = Path(self.store.parent, new_table_name)

        self.store = old_store.rename(new_store)

    def update_name(self, name=None, call_set=True):
        if name is None:
            name = self.text(0)
        self.adata.uns["tree_attrs"]["name"] = name
        if call_set:
            self.set_adata(self.adata)

    def update_children(self, call_set=True):
        children = self.collect_children()
        if children != []:
            self.adata.uns["tree_attrs"]["children"] = [
                str(child.store) for child in children
            ]
        else:
            self.adata.uns["tree_attrs"]["children"] = []
        if call_set:
            self.set_adata(self.adata)

    def update_parent(self, call_set=True):
        parent_store = str(self.parent().store) if self.parent() else None
        self.adata.uns["tree_attrs"]["parent"] = parent_store
        if call_set:
            self.set_adata(self.adata)

    # NOTE: consider using @property
    def set_adata(self, adata):
        self.adata = adata
        self.repr_view = QLabel()
        self.repr_view.setText(self.__repr__())
        self.treeWidget().setItemWidget(self, 1, self.repr_view)
        adata_rep = str(self.adata).replace("\n", "\n\n")
        tooltip = f"""
             <div style="max-width: 600px;">
                {adata_rep}
            </div>
        """
        self.setToolTip(0, tooltip)
        self.setToolTip(1, tooltip)

    def add_child_store(self, child: AnnDataNodeQT):
        self.adata.uns["tree_attrs"]["children"].append(str(child.store))

    def get_clusters(self):
        return self.adata.obs[self.name].unique()

    def get_cluster_subset(self, label, index_only=False):
        if index_only:
            return self.adata[self.adata.obs[self.name] == label].obs.index
        else:
            return self.adata[self.adata.obs[self.name] == label].copy()

    def collect_parents(self):
        """Collects all immediate and distant parent objects"""
        collection = []
        current_node = self
        while current_node.parent() is not None:
            current_node = current_node.parent()
            collection.append(current_node)
        return collection

    def collect_children(self):
        """Collects all immediate child objects"""
        n_children = self.childCount()
        collection = []
        for n in range(n_children):
            collection.append(self.child(n))
        return collection

    def collect_all_children(self):
        """Collects all child objects"""
        collection = []
        for child in self.collect_children():
            collection.append(child)
            collection.extend(child.collect_all_children())
        return collection

    # Directive -> Remerge on new obs
    def inherit_child_obs(self, child, log_steps) -> None:
        if log_steps:
            logging.debug("%s inheriting %s", self.text(0), child.text(0))

        parent_obs = self.adata.obs
        child_obs = child.adata.obs
        # check new cols, append with label if needed
        # If "new" is already in column from a different node of the same level,
        # Must ensure the columns are unique in a given subset.
        new_cols = list(set(child_obs.columns) - set(parent_obs.columns))
        if new_cols:
            parent_obs = parent_obs.merge(
                child_obs[new_cols],
                how="left",
                left_index=True,
                right_index=True,
            )

        # Check for nan columns to fill raggedly
        has_na = parent_obs.columns[parent_obs.isna().any()]
        if has_na.any():
            for c in has_na:
                s = parent_obs[c]
                na_indices = s[s.isna()].index

                if (
                    c in child_obs.columns
                    # Check that parent is indeed a full subset of child
                    and len(na_indices) >= child_obs.shape[0]
                    # Check that all new values in subset cover NaNs only
                    and all(child_obs.index.isin(na_indices))
                ):
                    # Reset cats
                    to_add = child_obs[c].copy().astype(str)
                    parent_obs[c] = parent_obs[c].astype(str)
                    parent_obs.loc[na_indices, c] = to_add
                    parent_obs[c] = parent_obs[c].astype("category")

        self.adata.obs = parent_obs

    def absorb_child_obs(self, child, log_steps) -> None:
        self.inherit_child_obs(child, log_steps)
        # self.removeChild(child) # We may want to keep the subsets ...

    def inherit_children_obs(self, log_steps=False) -> None:
        """Preorder Traversal"""
        # Traverse each child,
        children = self.collect_all_children()
        if len(children) > 0:
            for child in children:
                # Base case
                if child.childCount() == 0:
                    # Up/backpropagation
                    self.absorb_child_obs(child, log_steps)

                else:
                    child.inherit_children_obs(log_steps)
                    # After inheriting, if empty, then add
                    # self.inherit_children_obs()
                    self.absorb_child_obs(child, log_steps)

import logging
import re

import pandas as pd
from anndata import AnnData
from qtpy.QtWidgets import (
    QLabel,
    QTreeWidgetItem,
)


# QTree Versions
class AnnDataNodeQT(QTreeWidgetItem):
    def __init__(self, adata, labels, name, parent):
        """
        adata : AnnData
        labels : list of new cluster labels
        name : str, name of the cluster column
        parent : QTreeWidgetItem | QTreeWidget | None
        """
        super(QTreeWidgetItem, self).__init__(parent)
        self.setText(0, name)

        self.adata = adata
        self.labels = labels
        if labels is not None:
            if not isinstance(self.labels[0], str):
                self.labels = [str(x) for x in self.labels]

            if adata is not None:
                assert len(labels) == adata.shape[0]
                self.adata.obs[name] = labels

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

    # Model properties / functionality
    def set_adata(self, adata):
        self.adata = adata

    def get_clusters(self):
        return self.adata.obs[self.name].unique()

    def get_cluster_subset(self, label, index_only=False):
        if index_only:
            return self.adata[self.adata.obs[self.name] == label].obs.index
        else:
            return self.adata[self.adata.obs[self.name] == label].copy()

    def collect_child_adatas(self) -> list[AnnData]:
        n_children = self.childCount()
        collection = []
        for n in range(n_children):
            collection.append(self.child(n))
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
        new_cols = set(child_obs.columns) - set(parent_obs.columns)
        rename = {}
        node_label = child.text(0)
        for k in new_cols:
            rename[k] = node_label + "->" + k  # + "_" + node_label

        merged_obs = pd.merge(
            parent_obs, child_obs.rename(columns=rename), how="left"
        )

        self.adata.obs = merged_obs

    def absorb_child_obs(self, child, log_steps) -> None:
        self.inherit_child_obs(child, log_steps)
        # self.removeChild(child) # We may want to keep the subsets ...

    def inherit_children_obs(self, log_steps=False) -> None:
        """Preorder Traversal"""
        # Traverse each child,
        children = self.collect_child_adatas()
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

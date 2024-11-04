from typing import List
from anndata import AnnData
import pandas as pd
import scanpy as sc
import re
from PyQt5.QtWidgets import QTreeWidgetItem, QTreeWidget, QLabel, \
    QMenu, QFileDialog, QMessageBox

class AnnDataNode():
    def __init__(
        self, 
        adata=None, 
        labels=None, 
        name='root', # Column cluster name
        children=None):
        
        self.adata = adata
        self.shape = adata.shape
        self.labels = labels
        

        if labels is not None:
            if not isinstance(self.labels[0], str):
                self.labels = [str(l) for l in self.labels]
            
            if adata is not None:
                assert len(labels) == adata.shape[0]
                self.adata.obs[name] = labels

        self.name = name
        self.children = []

        if children is not None:
            for child in children:
                assert isinstance(child, AnnDataClusterNode)
                self.add_child(child)

    def __getattr__(self, attr):
        NODE_ATTRS = ["adata", "shape", "labels", "name", "children", "parent"]
        if attr in NODE_ATTRS:
            return self.__dict__[attr]
        else:
            return getattr(self.adata, attr)
        
    def __repr__(self, level=0, is_last=False):
        def remove_after_n_obs_n_vars(input_string):
            if input_string is None:
                return input_string
            else:
                pattern = r"(n_obs\s+×\s+n_vars\s+=\s+\d+\s+×\s+\d+)"
                match = re.search(pattern, input_string)
                if match:
                    return input_string[:match.end()]
                return input_string
            
        prefix = "    " * level + ("└── " if is_last else "├── " if level > 0 else "")
        out_repr = f"{prefix}{self.name} : {remove_after_n_obs_n_vars(str(self.adata))}"
        if self.children:
            for i, c in enumerate(self.children):
                is_last_child = (i == len(self.children) - 1)
                out_repr += "\n" + c.__repr__(level + 1, is_last_child)
        return out_repr
    
    def set_parent(self, parent):
        self.parent = parent

    def add_child(self, node):
        assert isinstance(node, AnnDataClusterNode)
        if node.name in [c.name for c in self.children]:
            raise ValueError(f"Node with the name {node.name } already exists. ")
        self.children.append(node)

    def annotate_node(self, label_map):
        """ label_map -> dict: {old_label: new_label} """
        assert len(set(self.labels)) == len(label_map)
        self.labels = [label_map[l] for l in self.labels]
        self.adata.obs[self.name] = self.labels
        
    def dotplot(self, **kwargs):
        """ Matrixplot / Dotplot to visualise clusters. """
        sc.pl.dotplot(
            self.adata, 
            self.adata.var_names,
            self.name,
            **kwargs
            )
        
    def matrixplot(self, **kwargs):
        """ Matrixplot / Dotplot to visualise clusters. """
        sc.pl.matrixplot(
            self.adata, 
            self.adata.var_names,
            self.name,
            **kwargs
            )
    
    def get_clusters(self):
        return self.adata.obs[self.name].unique()
    
    def get_cluster_subset(self, label, index_only=False):
        if index_only:
            return self.adata[self.adata.obs[self.name] == label].obs.index
        else:
            return self.adata[self.adata.obs[self.name] == label]

class AnnDataTree():
    def __init__(self, adata):
        # Initialise Tree
        self.root = AnnDataNode(adata=adata, name="root") # Nodes are 'rounds' in this case
    
    def __repr__(self):
        return repr(self.root)

    def __getitem__(self, node_name):
        return self.query_node(self.root, node_name)

    def query_node(self, node, node_name):
        """ Assumes unique node names. otherwise, returns first instance. """
        if node.name == node_name:
            return node
        
        else:
            for child in node.children:
                result = self.query_node(child, node_name)
                
                if result is not None:
                    return result
                else:
                    continue

            return None
        
    def add_round(self, adata_sub, labels, round_name, parent=None):
        # Add a clustering round to a given level in the tree
        # Query parent node 
        parent_node = self.query_node(self.root, parent)
        if parent_node is None:
            # set it to root. 
            parent_node = self.root

        parent_node.add_child(AnnDataClusterNode(adata_sub, labels, round_name))
        parent_node.children[-1].set_parent(parent_node)

    def get_cluster_subset(
            self, 
            node_label,
            cluster_subset=None,
            var_list=None):
        """ Get the cluster subset of a given node. If vars is not None,
            subsets different marker list than from the given node. Subsets from
            root anndata. """

        node = self.query_node(self.root, node_label)
        if var_list is not None:
            assert all(v in self.root.adata.var_names for v in var_list)

            if cluster_subset is not None:
                return self.root.adata[
                    node.get_cluster_subset(cluster_subset, index_only=True)][:, var_list]
                
            else:
                return self.query_node(self.root, node_label).adata[:, var_list]

        else:
            if cluster_subset is not None:
                return node.get_cluster_subset(cluster_subset)
            else:
                raise ValueError("Cannot subset clusters without specifying vars.")

    def _repr_html(self):
        """ Get rich html of AnnData subset. """
        pass

    def consolidate(self, node=None):
        """ Consolidate all the nodes into the root anndata. """
        # TODO: Things to track:
        # 1) subset relationships -> obs category + obs shape cells
        # 2) subset markers for a node -> log what markers were used for each node
        # X) or maybe log the tree structure somehow in a writable format on the AnnData

        if node is None:
            parent_node = self.root
        else:
            parent_node = node
        # Traverse each node, add the obs label to the root anndata
        for node in parent_node.children:
            if len(node.children) > 0:
                self.consolidate(node)

            # Ensure obs labels is proper
            index_to_label = dict(node.obs[node.name])
            self.root.obs[node.name] = self.root.obs.index.map(index_to_label)


# adata=None, 
#         labels=None, 
#         name='root', 
#         children=None):

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
                self.labels = [str(l) for l in self.labels]
            
            if adata is not None:
                assert len(labels) == adata.shape[0]
                self.adata.obs[name] = labels

        self.repr_view = QLabel()
        self.repr_view.setText(self.__repr__())
        self.treeWidget().setItemWidget(self, 1, self.repr_view)
        adata_rep = str(self.adata).replace("\n", "\n\n")
        tooltip = f'''
             <div style="max-width: 600px;">
                {adata_rep}
            </div>
        '''
        self.setToolTip(0, tooltip)
        self.setToolTip(1, tooltip)

        # self.setContextMenuPolicy(3)
        # self.customContextMenuRequested.connect(self.show_context_menu)

    # def __getattr__(self, attr):
    #     NODE_ATTRS = ["adata", "shape", "labels", "name", "children", "parent"]
    #     if attr in NODE_ATTRS:
    #         return self.__dict__[attr]
    #     else:
    #         try:
    #             return getattr(self.adata, attr)
    #         except AttributeError:
    #             return getattr(self, attr)
            
    def __repr__(self):
        def remove_after_n_obs_n_vars(input_string):
            if input_string is None:
                return input_string
            else:
                pattern = r"(n_obs\s+×\s+n_vars\s+=\s+\d+\s+×\s+\d+)"
                match = re.search(pattern, input_string)
                if match:
                    return input_string[:match.end()]
                return input_string
            
        out_repr = f"{remove_after_n_obs_n_vars(str(self.adata))}"

        return out_repr
    
    # Widget properties / functionality
    def show_context_menu(self, pos):
        item = self.itemAt(pos)
        if item is not None:  # Check if an item was clicked
            context_menu = QMenu(self)

            # Add actions to the context menu
            action_save = context_menu.addAction("Save")
            action_remove = context_menu.addAction("Delete")

            # Connect actions to their respective methods
            action_save.triggered.connect(
                lambda: self.save_item(item))
            
            action_remove.triggered.connect(
                lambda: self.remove_item(item))

            # Show the context menu
            context_menu.exec_(self.viewport().mapToGlobal(pos))

    def save_item(self, item):
        fname, _ = QFileDialog.getSaveFileName(
            self, "Write to disk", "", "")
        if fname:
            try:
                print(f"Saving to {fname}") # save_singular_..
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")

    def remove_item(self, item):
        print(f"mock Removing {item}")

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
    
    def collect_child_adatas(self) -> List[AnnData]:
        n_children = self.childCount()
        collection = []
        for n in range(n_children):
            collection.append(self.child(n))
        return collection

    # Directive -> Remerge on new obs
    def inherit_child_obs(self, child) -> None:
        parent_obs = self.adata.obs
        child_obs = child.adata.obs
        # check new cols, append with label if needed
        new_cols = set(child_obs.columns) - set(parent_obs.columns)
        rename = dict()
        node_label = child.text(0)
        for k in new_cols:
            rename[k] = node_label + "->" + k #+ "_" + node_label
        
        merged_obs = pd.merge(
            parent_obs, child_obs.rename(columns=rename), how="left")
        
        self.adata.obs = merged_obs

    def absorb_child_obs(self, child) -> None:
        self.inherit_child_obs(child)
        self.removeChild(child) # We may want to keep the subsets ... 

    def inherit_children_obs(self) -> None:
        # Traverse each child,
        for child in self.collect_child_adatas():
            assert child.childCount() >= 0 # Bound

            # Base case
            if child.childCount() == 0:
                print(f"{self.text(0)} <- inherit <- {child.text(0)}")
                # Up/backpropagation
                self.absorb_child_obs(child)

            # Queueing/DFS
            else:
                print(f"{self.text(0)} -> pass -> {child.text(0)}")
                child.inherit_children_obs()
                # After inheriting, if empty, then add
                if child.childCount() == 0:
                    self.inherit_children_obs()
                
    # # TODO; most important
    # def save_to_singular_adata(self, filepath) -> AnnData:
    #     self.inherit_children_obs()
    #     self.adata.write_h5ad(filepath)


    
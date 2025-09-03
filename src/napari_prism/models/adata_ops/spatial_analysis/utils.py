"""Functions for computing metrics from graph adjacency matrices contained
with anndata objects.

Some functions are wrappers for functions in scimap and squidpy, but
are implemented with subsetting to account for spatially distinct groups of
cells (i.e. cells belonging to different TMA cores, images, etc.)
"""

from time import time

import numpy as np
import scipy


def timer_func(func):
    # This function shows the execution time of
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"Function {func.__name__!r} executed in {(t2-t1):.4f}s")
        return result

    return wrap_func


def symmetrise_graph(adjacency_matrix):
    """Symmetrise a graph adjacency matrix"""
    # Symmetrize graph (make undirected); if A->B, then enforce B->A
    # Rather than dividing by 2, clip and sign to enforce 0/1
    if isinstance(adjacency_matrix, np.ndarray):
        sym = adjacency_matrix + adjacency_matrix.T
        sym = np.clip(sym, 0, 1)

    elif isinstance(adjacency_matrix, scipy.sparse.csr.csr_matrix):
        sym = adjacency_matrix + adjacency_matrix.T
        sym = sym.sign()

    else:
        raise ValueError("invalid adjacency matrix type")

    return sym


def normalise_log2p(X, pseudocount=1e-3):
    """Given a df, re-normalise dfm apply log2p with pseudocount."""
    X = X.div(X.sum(axis=1), axis=0)  # Normalise rows to add to 1
    return X.map(
        lambda x: np.log2(x + pseudocount)
    )  # Apply log2p transformations to every value


def annotate_tree(
    start_node,
    data,
    node_label_A=None,
    node_label_B=None,
):
    """Creates or annotates a leaf with forks from a start_node, then annotates
    the tuple-keyed node with data. Non-recursive to enforce explicit node
    traversals.

    If node_label_A and node_label_B is supplied:

                    start_node
                    ||      ||
                node_label_A  etc
                ||        ||
           node_label_B  etc
                ||
               data

    If node_label_A is supplied but not node_label_B, annotates
    data.
                    start_node
                    ||       ||
               node_label_A  etc
                    ||
                   data

    Vice-versa if node_label_A is not supplied:
                    start_node
                    ||       ||
              node_label_B  etc
                    ||
                   data

    If no node_labels supplied it will annotate the start_node with data
    directly:
                    start_node
                    ||      ||
                   data     etc

    If data is a dict, then the node reference at data is returned.
    """
    node = start_node

    # Keep track of parent and key for updating in place
    parent = None
    key = None

    if node_label_A:
        if node_label_A not in node:
            node[node_label_A] = {}
        parent, key = node, node_label_A
        node = node[node_label_A]  # Terminate at node_label

        if node_label_B:
            if node_label_B not in node:
                node[node_label_B] = {}
            parent, key = node, node_label_B
            node = node[node_label_B]  # Terminate at node_label -> p1/p2

    else:
        if node_label_B:  # no node_label, but p1 and p2
            if node_label_B not in node:
                node[node_label_B] = {}
            parent, key = node, node_label_B
            node = node[node_label_B]

    # Update the parent node's key with data
    if parent is not None:
        if isinstance(data, dict):
            parent[key].update(data)
        else:
            parent[key] = data
    else:
        if isinstance(data, dict):
            start_node.update(data)
        else:
            start_node["value"] = data

    if isinstance(node, dict):
        return node

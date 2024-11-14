"""Tests the subsetter module, or the used QT AnnData Trees"""

import logging
from typing import Any

import numpy as np


def test_inherit_children_obs(adata_tree_widget_populated: Any, caplog: Any):
    EXPECTED_TRAVERSAL_ORDER = [
        "Root inheriting 100",  # Inherits left node first since its terminal
        "50 inheriting 50_HES4_TNFRSF4",  # Recurse right subtree
        "50_SSU72_PARK7 inheriting TERMINAL",  # Terminal of right subtrees
        "50 inheriting 50_SSU72_PARK7",  # Begin back propagation from terminal node
        "Root inheriting 50",  # Back propagate to root node
    ]

    EXPECTED_NEW_COLS = [
        "100->100_annotation",
        "50->50_HES4_TNFRSF4->HES4_annotation",
        "50->50_SSU72_PARK7->SSU72_annotation",
        "50->50_SSU72_PARK7->TERMINAL->TERMINAL_annotation",
    ]

    root_node = adata_tree_widget_populated.topLevelItem(0)
    with caplog.at_level(logging.DEBUG):
        root_node.inherit_children_obs(log_steps=True)

    # 1) Check that the the preorder traversal order is correct
    log_text = caplog.text
    last_index = -1
    for log in EXPECTED_TRAVERSAL_ORDER:
        current_index = log_text.index(log)
        assert current_index > last_index, "Out of order"
        last_index = current_index

    # 2) and that all of the obs have been propagated up to the root node
    root_obs = root_node.adata.obs
    assert all(c in root_obs.columns for c in EXPECTED_NEW_COLS)

    for c in EXPECTED_NEW_COLS:
        vc = root_obs[c].value_counts(dropna=False)
        assert np.nan in vc
        assert vc[np.nan] == 650  # pbmcreduced -> 700, - 50 assigned

"""ML Models and statistical methods for characterising discrete labels."""

from typing import Literal, Union

import pandas as pd
from anndata import AnnData


# General Applicable Functions
# Binary Labels
# Univariate -> T-Test / Wilcoxon Rank Sum / Mann Whitney U
def univariate_test_feature_by_binary_label(
    patient_adata: AnnData,
    feature_column: str,
    label_column: str,
    parametric: bool = True,
    equal_var: bool = True,
    test: Literal["student_t", "welch_t", "wilcoxon", "mann-whitney"] = None,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Perform a univariate test to compare the distribution of a feature
    between two groups.

    Args:
        patient_adata (AnnData): AnnData object where rows represent patients or
            samples.
        feature_column (str): Column name in .obs of the feature to test.
        label_column (str): Column name in .obs of the binary label.
        parametric (bool, optional): Whether to use a parametric test. If True,
            uses a t-test, otherwise uses a mann-whitney U test. Defaults to
            True.
        equal_var (bool, optional): Whether to assume equal variance. Defaults
            to True. If False, overrides parametric test to perform a Welch's
            t-test.
        test (Literal["t-test", "wilcoxon", "mann-whitney"], optional):
            Manually specify the test to use. Overrides selections made from
            parameters. Defaults to None.
    """


def dom_feature_by_binary_label(
    patient_adata: AnnData,
    feature_column: str,
    label_column: str,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Calculate the difference of means between two groups."""


# Multivariate -> Logistic Regression


# Specialised Functions
def cellular_neighborhood_enrichment():
    pass

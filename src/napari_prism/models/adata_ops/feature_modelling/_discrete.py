"""ML Models and statistical methods for characterising discrete labels."""

from typing import Literal, Union

import numpy as np
import pandas as pd
import scipy.stats as scstats
import statsmodels.api as sm
from anndata import AnnData
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from napari_prism.models.adata_ops.feature_modelling._obs import ObsAggregator


# Helpers
def _build_design_matrix(Y_cond, X_cond, G, neighborhood, ct):
    """Builds a design matrix based on multiple conditional probabilities,
    given a `neighborhood` and `ct`,

    Y_cond -> Y(sample, n, ct | n = neighborhood, ct = ct)
    X_cond -> X(ct | ct = ct)
    G -> G(s | s = s)

    Return dataframe of columns: [Y_cond, X_cond, G]
    with rows as samples with non-na values for all columns (rows or sample
    with missing values are omitted from design matrix).

    """
    G_drop_na = G.dropna()
    Y_cond_cols = Y_cond.columns
    Y_col_sub = Y_cond_cols.get_level_values(level=1).get_loc(ct)
    Y_sub = Y_cond.xs(neighborhood, level=0)[Y_cond_cols[Y_col_sub]]
    Y_sub = Y_sub.loc[G_drop_na.index]
    Y_sub_remove_nan = Y_sub.dropna()
    # And dont include samples that dont have a value for the response variable
    Y_sub_remove_nan.name = f"Y(sample, ct, n | ct = {ct}, n = {neighborhood})"

    common_index = Y_sub_remove_nan.index
    # Get X(ct = CT), which are the transformed CT frequencies per sample
    X_cond = X_cond.loc[G_drop_na.index]
    X_cond_cols = X_cond.columns
    X_col_sub = X_cond_cols.get_level_values(level=1).get_loc(ct)
    X_sub = X_cond.loc[common_index, X_cond_cols[X_col_sub]]
    X_sub.name = f"X(sample, ct | ct = {ct})"

    # Get G(sample)
    G_sub = G.loc[common_index]
    # Get F(sample) --> all are 1..? NOTE: this maybe redundant then
    F_sub = pd.Series(np.ones(len(G_sub)))
    F_sub.index = common_index
    F_sub.name = "F"
    # contrsuct design matrix for logging
    design_df = pd.concat([Y_sub_remove_nan, X_sub, G_sub, F_sub], axis=1)

    return design_df


def _consolidate_statistics(results, rownames, colnames):
    """Given the output results from the linear model loops, organise into a matrix:
    Neighborhoods as rows,
    Phenotypes as columns,
    values as p-values / coefficients.

    All are respective to 1/0
    """
    index = pd.MultiIndex.from_tuples(
        results.keys(), names=[colnames, rownames]
    )
    df = pd.DataFrame(list(results.values()), index=index).unstack(level=0)
    df.columns = df.columns.droplevel(0)
    return df


def _normalise_log2p(X, pseudocount=1e-3):
    """Given a df, re-normalise dfm apply log2p with pseudocount."""
    X = X.div(X.sum(axis=1), axis=0)  # Normalise rows to add to 1
    return X.map(
        lambda x: np.log2(x + pseudocount)
    )  # Apply log2p transformations to every value


def _get_x_y_by_binary_label(
    patient_adata: AnnData,
    feature_column: str | list[str],
    label_column: str,
    attr: Literal["obs", "X"] = "obs",
    fillna_value: float = 0.0,
) -> dict[str, Union[pd.Series, pd.DataFrame, np.ndarray]]:
    if attr == "X":
        feature_index = patient_adata.var_names.get_loc(feature_column)
        feature_X = pd.DataFrame(patient_adata.X[:, feature_index])

    elif attr == "obs":
        feature_X = patient_adata.obs[feature_column]

    else:
        raise ValueError("Attribute must be either 'obs' or 'X'.")

    # validate columns as numeric
    assert pd.api.types.is_numeric_dtype(
        feature_X
    ), "Feature column must be numeric."

    label_y = patient_adata.obs[label_column]

    # Filter out NAN labels;
    non_na_indices = label_y.isna()[~label_y.isna()].index
    label_y = label_y[non_na_indices]

    # Filter out NAN features, or fillnas.
    non_na_feature_indices = feature_X[~feature_X.isna()].index
    feature_X = feature_X.loc[non_na_feature_indices]

    # Assert binary label;
    if not isinstance(label_y, bool):
        assert len(label_y.unique()) == 2, "Label column must be binary."

    conditions = label_y.unique()
    cond1_indices = label_y[label_y == conditions[0]].index
    cond2_indices = label_y[label_y == conditions[1]].index
    cond1_X = feature_X[cond1_indices]
    cond2_X = feature_X[cond2_indices]

    return {conditions[0]: cond1_X, conditions[1]: cond2_X}


# General Applicable Functions
# Binary Labels
# Univariate -> T-Test / Wilcoxon Rank Sum or Mann Whitney U
def difference_of_means_by_binary_label(
    patient_adata: AnnData,
    feature_column: str,  # | list[str],
    label_column: str,
    attr: Literal["obs", "X"] = "obs",
    fillna_value: float = 0.0,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Calculate the difference of means between two groups.

    Args:
        patient_adata (AnnData): AnnData object where rows represent patients or
            samples.
        feature_column (str): Column index in .var of the feature to test.
        label_column (str): Column name in .obs of the binary label.
    """
    groups = _get_x_y_by_binary_label(
        patient_adata, feature_column, label_column, attr, fillna_value
    )
    cond1, cond2 = list(groups.keys())
    cond1_X = groups[cond1]
    cond2_X = groups[cond2]

    cond1_X_mean = cond1_X.mean()
    cond2_X_mean = cond2_X.mean()

    cond1_X_std = cond1_X.std()
    cond2_X_std = cond2_X.std()

    cond1_n = cond1_X.shape[0]
    cond2_n = cond2_X.shape[0]

    cond1_ci = 1.96 * cond1_X_std / np.sqrt(cond1_n)
    cond2_ci = 1.96 * cond2_X_std / np.sqrt(cond2_n)

    diff_means = cond1_X_mean - cond2_X_mean
    diff_ci = np.sqrt(cond1_ci**2 + cond2_ci**2)

    return diff_means, diff_ci


def univariate_test_feature_by_binary_label(
    patient_adata: AnnData,
    feature_column: str,  # | list[str],
    label_column: str,
    attr: Literal["obs", "X"] = "obs",
    parametric: bool = True,
    equal_var: bool = True,
) -> Union[pd.DataFrame, pd.DataFrame]:
    """Perform a univariate test to compare the distribution of a feature
    between two groups.

    Args:
        patient_adata (AnnData): AnnData object where rows represent patients or
            samples.
        feature_column (str): Column index in .var or column name in .obs to
            test. Must be numerical.
        label_column (str): Column name in .obs of the binary label.
        attr (str): Attribute in the AnnData object to extract the features
            from.
        parametric (bool, optional): Whether to use a parametric test. If True,
            uses a t-test, otherwise uses a mann-whitney U test. Defaults to
            True.
        equal_var (bool, optional): Whether to assume equal variance. Defaults
            to True to perform Student's t-test. If False, perform a Welch's
            t-test.
    """
    groups = _get_x_y_by_binary_label(
        patient_adata,
        feature_column,
        label_column,
        attr,
    )
    cond1, cond2 = list(groups.keys())
    cond1_X = groups[cond1]
    cond2_X = groups[cond2]

    test = scstats.ttest_ind if parametric else scstats.mannwhitneyu

    kwargs = {}
    if parametric:
        kwargs["equal_var"] = equal_var

    statistic, p_value = test(cond1_X, cond2_X, **kwargs)

    return statistic, p_value


# Multivariate -> Logistic Regression
def logistic_regression_by_binary_label(
    patient_adata: AnnData,
    feature_column: str | list[str],
    label_column: str,
    fillna_value: float = 0.0,
    balance_labels: bool = True,
    penalty: Literal["l1", "l2", "elasticnet"] = "l2",
    regularization_strength: float = 0.0,
) -> Union[pd.DataFrame, pd.DataFrame]:
    class_weight = None
    if balance_labels:
        class_weight = "balanced"

    inv_reg_strength = 1.0 - regularization_strength

    logreg = LogisticRegression(  # noqa: F841
        class_weight=class_weight, C=inv_reg_strength
    )


def logistic_regression_by_multiclass_label(
    patient_adata: AnnData,
    feature_column: str | list[str],
    label_column: str,
    fillna_value: float = 0.0,
) -> Union[pd.DataFrame, pd.DataFrame]:
    pass


# Specialised Functions
# NOTE: can generalise 'neighborhod' to 'compartment'
def cellular_neighborhood_enrichment(
    adata: AnnData,
    neighborhood: str,
    phenotype: str,
    label: str,
    grouping: str,
    pseudocount: float = 1e-3,
    multiple_testing_correction: Literal[
        "bonferroni", "fdr_bh", "hommel", "holm"
    ] = "bonferroni",
) -> dict:
    """
    Perform a cellular neighborhood enrichment test using OLS linear models.

    Args:
        adata: Annotated data object.
        neighborhood: Column in .obs that defines the neighborhood index
            or label that a cell belongs to.
        phenotype: Column in .obs that defines the cellular label to take
            into account. Ideally this should be the phenotype that was used to
            compute the given `neighborhood`.
        label: Column in .obs that defines the binary label defining
            distinct `grouping` groups.
        grouping: Column in .obs that defines distinct samples.
        pseudo_count: Pseudocount to add to the log2
            normalised proportions data. Defaults to 1e-3.

    Returns:
        Dictionary containing the p-values, adjusted p-values,
            coefficients, t-values and null hypothesis rejection status.
            Each entry is a DataFrame with neighborhoods as rows and
            phenotypes as columns.
    """
    unique_phenotypes = adata.obs[phenotype].unique()
    unique_neighborhoods = adata.obs[neighborhood].unique()

    nhood_agg = ObsAggregator(adata, [neighborhood, grouping])
    sample_agg = ObsAggregator(adata, grouping)
    sample_to_response = sample_agg.get_metadata_df(label)

    # PNC 2D matrix
    ct_props_by_sample = sample_agg.get_category_proportions(phenotype)

    ct_props_by_neighborhood_and_sample = nhood_agg.get_category_proportions(
        phenotype
    )

    X_sample_log2 = _normalise_log2p(ct_props_by_sample)
    X_sample_neighborhood_log2 = ct_props_by_neighborhood_and_sample.groupby(
        level=0, group_keys=False
    ).apply(_normalise_log2p, pseudocount)

    design_matrices = {}
    p_values = {}
    coefficients = {}
    t_values = {}
    label_encodings = {}

    encoder = LabelEncoder()
    for ct in unique_phenotypes:
        for n in unique_neighborhoods:
            design_df = _build_design_matrix(
                X_sample_neighborhood_log2,
                X_sample_log2,
                sample_to_response,
                n,
                ct,
            )
            Y = design_df.iloc[:, 0]
            X = design_df.iloc[:, 1:]
            X[label] = encoder.fit_transform(X[label].values.reshape(-1, 1))
            regress = sm.OLS(Y, X).fit()
            p_values[(ct, n)] = regress.pvalues[label]
            coefficients[(ct, n)] = regress.params[label]
            t_values[(ct, n)] = regress.tvalues[label]
            design_matrices[(ct, n)] = design_df
            label_encodings[(ct, n)] = dict(enumerate(encoder.classes_))

    p_values_df = _consolidate_statistics(p_values, neighborhood, phenotype)

    coefficients_df = _consolidate_statistics(
        coefficients, neighborhood, phenotype
    )

    t_values_df = _consolidate_statistics(t_values, neighborhood, phenotype)

    null_hypothesis, adjusted_pvalues, _, _ = sm.stats.multipletests(
        p_values_df.values.flatten(), method=multiple_testing_correction
    )

    null_hypothesis_df = pd.DataFrame(
        null_hypothesis.reshape(p_values_df.shape)
    )
    null_hypothesis_df.columns = p_values_df.columns
    null_hypothesis_df.index = p_values_df.index

    adjusted_pvalues_df = pd.DataFrame(
        adjusted_pvalues.reshape(p_values_df.shape)
    )
    adjusted_pvalues_df.columns = p_values_df.columns
    adjusted_pvalues_df.index = p_values_df.index

    results = {
        "p_values": p_values_df,
        "adjusted_p_values": adjusted_pvalues_df,
        "reject_null_hypothesis": null_hypothesis_df,
        "coefficients": coefficients_df,
        "t_values": t_values_df,
    }

    return results

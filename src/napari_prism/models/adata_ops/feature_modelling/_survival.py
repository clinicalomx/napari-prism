"""Models for time series (survival) data."""

# Utility
# Parse survival columns as a structure array for appropraite input
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from anndata import AnnData
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv

from napari_prism.models.adata_ops.feature_modelling._obs import ObsAggregator


def get_sample_level_adata(
    adata: AnnData,
    sample_column: str,
    feature_columns: list[str] | None = None,
) -> AnnData:
    """
    Return an AnnData object indexed by samples. If feature_columns is None,
    then obs returns all columns which are parallel keys (1to1) or super keys
    (Nto1) to `sample_column`.

    Args:
        adata: AnnData object.
        sample_column: Column name in .obs to use as the sample index.
        feature_columns: List of column names in .obs to use as features.
            If None, then all columns which are parallel keys (1to1) or super
            keys (Nto1) to `sample_column` are used.

    Returns:
        AnnData object indexed by samples (i.e. a patient level AnnData).
    """
    agg = ObsAggregator(adata, sample_column)
    if feature_columns is None:
        df = agg.groupby_df[agg.parallel_keys].first()
        return AnnData(obs=df)
    else:
        raise NotImplementedError()


def parse_survival_columns(
    adata: AnnData,
    event_column: str,
    time_column: str,
) -> np.ndarray:
    surv_parser = Surv()
    return surv_parser.from_arrays(
        adata.obs[event_column], adata.obs[time_column]
    )


# Univariate -> KM
def kaplan_meier(
    adata: AnnData,
    event_column: str,
    time_column: str,
    stratifier: str = None,
    **kwargs,
) -> np.ndarray:
    survival = parse_survival_columns(adata, event_column, time_column)

    results = {}
    if stratifier is not None:
        unique_labels = adata.obs[stratifier].unique()
        # Filter out nas;
        print(
            f"Filtering out NAs for factor {stratifier}: {adata.obs[stratifier].isna().sum()}"
        )
        unique_labels = [x for x in unique_labels if str(x) != "nan"]
        for g in unique_labels:
            g_mask = adata.obs[stratifier] == g
            results[g] = kaplan_meier_estimator(
                survival[g_mask.values]["event"],
                survival[g_mask.values]["time"],
                conf_type="log-log",
                time_enter=np.zeros_like(survival[g_mask.values]["time"]),
                **kwargs,
            )
    else:
        results["all"] = kaplan_meier_estimator(
            survival["event"], survival["time"], conf_type="log-log"
        )
    return results


# def count_patients_per_time_step(
#     adata: AnnData,
#     time_column: str,
#     time_steps: list[float | int],
#     stratifier: str = None,
# ) -> pd.DataFrame:
#     pass


def plot_kaplan_meier(
    km_dict,
    with_counts=False,
    stratifier_colors=None,
    xlabel=None,
    ylabel=None,
    fill_alpha=0.05,
    *,
    adata=None,
    event_column=None,
    time_column=None,
    stratifier=None,
):
    keys = km_dict.keys()
    counts = {}

    bottomoffset = -0.175
    koffset = 0
    # secondx_offset = -0.5

    max_fives = max([max(km_dict[k][0]) for k in keys])
    max_fives = math.ceil(max_fives / 5) * 5
    time_steps = range(0, max_fives + 1, 5)

    fig, ax = plt.subplots()
    secondx_created = False
    for k in keys:
        style = {}
        if stratifier_colors is not None:
            style["color"] = stratifier_colors[k]

        time, sprob, conf_int = km_dict[k]

        ax.step(time, sprob, where="post", label=f"{k}", **style)
        ax.fill_between(
            time,
            conf_int[0],
            conf_int[1],
            alpha=fill_alpha,
            step="post",
            **style,
        )
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0 - 1, max_fives + 1)
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if with_counts:
            assert adata is not None
            assert event_column is not None
            assert time_column is not None
            assert stratifier is not None
            # round to nearest 5
            # Then for each key, compute survival,
            g_mask = adata.obs[stratifier] == k
            survival = parse_survival_columns(adata, event_column, time_column)
            alive = np.zeros_like(time_steps)
            for i, t in enumerate(time_steps):
                for patient in survival[g_mask.values]:
                    e, d = patient
                    if (d > t) or (d == t) and (e is False):
                        alive[i] += 1
                    else:
                        continue
            counts[k] = alive

            # Get the x spine and duplicate it downwards

            # axouter = plt.axes([0,0,1,1], facecolor=(1,1,1,0))
            # At each time step, plot the number of patients alive; the number
            for i, c in enumerate(counts[k]):
                yval = koffset + bottomoffset

                plt.text(
                    time_steps[i] - 0.5,
                    yval,
                    f"{c}",
                    fontdict={"weight": "normal", "size": 10},
                )

            # plt.gca().add_line(line)

            koffset -= 0.065
        # Move the secondary x-axis 5 units down
        else:
            ax.set_xlabel(xlabel)

    if not secondx_created and with_counts:
        ax2 = ax.secondary_xaxis(yval - 0.025)
        ax2.set_xlabel(xlabel)
        secondx_created = True


# Multivariate -> Cox PH
def cox_proportional_hazard(
    adata: AnnData,
    event_column: str,
    time_column: str,
    covariates: list[str] | None,
    stratifier: str = None,
    l1_ratio: float = 1.0,
):
    survival = parse_survival_columns(adata, event_column, time_column)
    # TODO: Check numerical or categrical covariates in those given;
    features_X = adata.X if covariates is None else adata.obs[covariates]

    cox_instance = CoxnetSurvivalAnalysis()
    cox_instance.set_params(l1_ratio=l1_ratio)
    cox_instance.fit(features_X, survival)

    # Post
    concordance = cox_instance.score(features_X, survival)
    coefficients = pd.DataFrame(
        cox_instance.coef_,
        index=covariates if not None else adata.var_names,
        columns=np.round(cox_instance.alphas_, 10),
    )
    return concordance, coefficients, cox_instance


def plot_cox_coefficients(coefs, n_highlight=10):
    _, ax = plt.subplots(figsize=(9, 6))
    alphas = coefs.columns
    for row in coefs.itertuples():
        ax.semilogx(alphas, row[1:], ".-", label=row.Index)

    alpha_min = alphas.min()
    print(coefs.loc[:, alpha_min].map(abs))
    top_coefs = (
        coefs.loc[:, alpha_min].map(abs).sort_values().tail(n_highlight)
    )
    for name in top_coefs.index:
        coef = coefs.loc[name, alpha_min]
        plt.text(
            alpha_min,
            coef,
            name + "   ",
            horizontalalignment="right",
            verticalalignment="center",
        )

    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.grid(True)
    ax.set_xlabel("alpha")
    ax.set_ylabel("coefficient")


# Multivariate -> Random Survival Forest
def random_survival_forest(
    adata: AnnData,
    event_column: str,
    time_column: str,
    covariates: list[str] | None,
    stratifier: str = None,
    n_estimators: int = 100,
    min_samples_split: int = 10,
    min_samples_leaf: int = 15,
    n_jobs=-1,
):
    survival = parse_survival_columns(adata, event_column, time_column)
    features_X = adata.X if covariates is None else adata.obs[covariates]

    random_forest_instance = RandomSurvivalForest()
    random_forest_instance.set_params(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        n_jobs=n_jobs,
    )
    concordance = random_forest_instance.score(  # noqa: F841
        features_X, survival
    )


# Multivariate -> Gradiant Boosted
# Base Learner -> Regression Tree (versatile)
# Base Learner -> Component Wise (has feature selection)

# Multivariate -> SVM: Ranking problem -> Assign samples a rank based on survival time
# Multivariate -> SVM: Regression problem -> Predict survival time directly
# Kernel -> Linear
# Kernel -> RBF
# Kernel -> [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
# Kernel -> Clinical Kernel (custom)

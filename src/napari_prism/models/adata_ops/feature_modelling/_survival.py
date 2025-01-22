"""Models for time series (survival) data."""

# Utility
# Parse survival columns as a structure array for appropraite input
import numpy as np
from anndata import AnnData
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.util import Surv


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
) -> np.ndarray:
    survival = parse_survival_columns(adata, event_column, time_column)

    results = {}
    if stratifier is not None:
        unique_labels = adata.obs[stratifier].unique()
        for g in unique_labels:
            g_mask = adata.obs[stratifier] == g
            results[g] = kaplan_meier_estimator(
                survival[g_mask]["event"],
                survival[g_mask]["time"],
                conf_type="log-log",
            )
    else:
        results["all"] = kaplan_meier_estimator(
            survival["event"], survival["time"], conf_type="log-log"
        )
    return results


# Accompanying plot -> TODO move to plot / widget modules
import matplotlib.pyplot as plt  # noqa: E402


def plot_kaplan_meier(km_dict):
    keys = km_dict.keys()
    for k in keys:
        time, sprob, conf_int = km_dict[k]
        plt.step(time, sprob, where="post", label=f"{k}")
        plt.fill_between(
            time, conf_int[0], conf_int[1], alpha=0.2, step="post"
        )
        plt.ylim(0, 1)
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))


# Multivariate -> Cox PH
def cox_proportional_hazard(
    adata: AnnData,
    event_column: str,
    time_column: str,
    covariates: list[str],
    stratifier: str = None,
    regularization_strength: float = 0.0,
):
    # TODO: Check numerical or categrical covariates in those given;
    cox_instance = CoxPHSurvivalAnalysis(  # noqa: F841
        alpha=regularization_strength
    )


# Penalty -> L2/Ridge (Soft)
# Penalty -> L1/Lasso (Hard)

# Multivariate -> Random Survival Forest

# Multivariate -> Gradiant Boosted
# Base Learner -> Regression Tree (versatile)
# Base Learner -> Component Wise (has feature selection)

# Multivariate -> SVM: Ranking problem -> Assign samples a rank based on survival time
# Multivariate -> SVM: Regression problem -> Predict survival time directly
# Kernel -> Linear
# Kernel -> RBF
# Kernel -> [‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’, ‘laplacian’, ‘sigmoid’, ‘cosine’]
# Kernel -> Clinical Kernel (custom)

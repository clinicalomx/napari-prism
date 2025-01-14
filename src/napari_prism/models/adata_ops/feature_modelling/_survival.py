"""Models for time series (survival) data."""

# Utility
# Binner Method to discretise particular contiuous features as separable
# curves

# Univariate -> KM

# Multivariate -> Cox PH
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

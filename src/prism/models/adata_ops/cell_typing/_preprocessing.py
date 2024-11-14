from collections.abc import Callable
from functools import partial
from typing import Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.stats as stats
from anndata import AnnData
from joblib import Parallel, delayed


def filter_by_obs_count(
    adata: AnnData,
    obs_col: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> AnnData:
    """Filters cells which belong to a category in `AnnData.obs` with a cell
    count less than a `min_value` and/or more than a `max_value`. If layer is
    None, does this on .X.

    Args:
        adata: Anndata object.
        obs_col: `AnnData.obs` column to filter by.
        min_value (optional): Minimum value to filter by.
        max_value (optional): Maximum value to filter by.

    Returns:
        Filtered AnnData object.
    """
    MIN_DEFAULT = 10
    MAX_DEFAULT = 1000000

    if not isinstance(
        adata.obs[obs_col].dtype, pd.CategoricalDtype
    ) or pd.api.types.is_numeric_dtype(adata.obs[obs_col]):
        raise ValueError("Must be a categorical .obs column.")

    cells_by_obs = adata.obs[obs_col].value_counts()
    if min_value is None and max_value is None:
        print("Min and max value not specified, filtering by defaults")

        print(f"Excluding {obs_col} with less than {MIN_DEFAULT} cells")
        min_value = MIN_DEFAULT

        if adata.shape[0] < MAX_DEFAULT:
            print(f"Excluding {obs_col} with more than {MAX_DEFAULT} cells")
            max_value = MAX_DEFAULT

    if min_value is not None:
        cells_by_obs = cells_by_obs[cells_by_obs > min_value]

    if max_value is not None:
        cells_by_obs = cells_by_obs[cells_by_obs < max_value]

    return adata[adata.obs[obs_col].isin(cells_by_obs.index)]


def filter_by_obs_value(
    adata: AnnData,
    obs_col: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.obs` column less
    than a `min_value` and/or more than a `max_value`.

    Args:
        adata: Anndata object.
        obs_col: `AnnData.obs` column to filter by.
        min_value (optional): Minimum value to filter by.
        max_value (optional): Maximum value to filter by.

    Returns:
        Filtered AnnData object.
    """

    MIN_DEFAULT = 10
    MAX_DEFAULT = 100000  # 255 8 bit, 1 pixel below

    if isinstance(
        adata.obs[obs_col].dtype, pd.CategoricalDtype
    ) or not pd.api.types.is_numeric_dtype(adata.obs[obs_col]):
        raise ValueError("Must be a numerical .obs column.")

    if min_value is None and max_value is None:
        print("Min and max value not specified, filtering by defaults")

        print(f"Excluding cells with {obs_col} values below {MIN_DEFAULT}")
        min_value = MIN_DEFAULT

        print(f"Excluding cells with {obs_col} values above {MAX_DEFAULT}")
        max_value = MAX_DEFAULT

    if min_value is not None:
        adata[adata.obs[obs_col] > min_value]

    if max_value is not None:
        adata[adata.obs[obs_col] < max_value]

    return adata


def filter_by_obs_quantile(
    adata: AnnData,
    obs_col: str,
    min_quantile: float | None = None,
    max_quantile: float | None = None,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.obs` column less
    than the `min_quantile` and/or more than the `max_quantile`.

    Args:
        adata: Anndata object.
        obs_col: `AnnData.obs` column to filter by.
        min_quantile (optional): Minimum quantile to filter by.
        max_quantile (optional): Maximum quantile to filter by.

    Returns:
        Filtered AnnData object.
    """
    MIN_DEFAULT = 0.1
    MAX_DEFAULT = 0.95

    if isinstance(
        adata.obs[obs_col].dtype, pd.CategoricalDtype
    ) or not pd.api.types.is_numeric_dtype(adata.obs[obs_col]):
        raise ValueError("Must be a numerical .obs column.")

    if min_quantile is None and max_quantile is None:
        print("Min and max value not specified, filtering by defaults")

        print(
            f"Excluding cells with {obs_col} below the {MIN_DEFAULT} percentile"
        )
        min_quantile = MIN_DEFAULT

        print(
            f"Excluding cells with {obs_col} above the {MAX_DEFAULT} percentile"
        )
        max_quantile = MAX_DEFAULT

    if min_quantile is not None:
        assert min_quantile >= 0 and min_quantile <= 1
        min_obs_val = np.quantile(adata.obs[obs_col], min_quantile)
        adata = adata[adata.obs[obs_col] > min_obs_val]

    if max_quantile is not None:
        assert max_quantile >= 0 and max_quantile <= 1
        max_obs_val = np.quantile(adata.obs[obs_col], max_quantile)
        adata = adata[adata.obs[obs_col] < max_obs_val]

    return adata


def filter_by_var_value(
    adata: AnnData,
    var: str,
    min_value: float | None = None,
    max_value: float | None = None,
    layer: str | None = None,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.var` column less
    than a `min_value` and/or more than a `max_value`.

    Args:
        adata: Anndata object.
        var: `AnnData.var` column to filter by.
        min_value (optional): Minimum value to filter by.
        max_value (optional): Maximum value to filter by.
        layer (optional): Expression layer to filter. Defaults to None (.X).

    Returns:
        Filtered AnnData object.
    """
    MIN_DEFAULT = 1  # 0 pixels above
    MAX_DEFAULT = 254  # 255 8 bit, 1 pixel below

    assert var in adata.var_names

    if min_value is None and max_value is None:
        print("Min and max value not specified, filtering by defaults")

        print(f"Excluding cells with {var} intensities below {MIN_DEFAULT}")
        min_value = MIN_DEFAULT

        print(f"Excluding cells with {var} intensities above {MAX_DEFAULT}")
        max_value = MAX_DEFAULT

    arr = adata[:, var].X if layer is None else adata[:, var].layers[layer]

    if min_value is not None:
        adata = adata[arr > min_value]

    if max_value is not None:
        adata = adata[arr < max_value]

    return adata


def filter_by_var_quantile(
    adata: AnnData,
    var: str,
    min_quantile: float | None = None,
    max_quantile: float | None = None,
    layer: str | None = None,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.var` column less
    than the `min_quantile` and/or more than the `max_quantile`.

    Args:
        adata: Anndata object.
        var: `AnnData.var` column to filter by.
        min_quantile (optional): Minimum quantile to filter by.
        max_quantile (optional): Maximum quantile to filter by.
        layer (optional): Expression layer to filter. Defaults to None (.X).

    """
    MIN_DEFAULT = 0.05
    MAX_DEFAULT = 0.99

    assert var in adata.var_names

    if min_quantile is None and max_quantile is None:
        print("Min and max value not specified, filtering by defaults")

        print(f"Excluding cells with {var} below the {MIN_DEFAULT} percentile")
        min_quantile = MIN_DEFAULT

        print(f"Excluding cells with {var} above the {MAX_DEFAULT} percentile")
        max_quantile = MAX_DEFAULT

    arr = adata[:, var].X if layer is None else adata[:, var].layers[layer]
    if min_quantile is not None:
        assert min_quantile >= 0 and min_quantile <= 1
        min_var_val = np.quantile(arr, min_quantile)
        adata = adata[arr > min_var_val]

    if max_quantile is not None:
        assert max_quantile >= 0 and max_quantile <= 1
        max_var_val = np.quantile(arr, max_quantile)
        adata = adata[arr < max_var_val]

    return adata


# TODO: Below class may be unnecessary / convert to static class
class AnnDataProcessor:
    """Processes an AnnData object. Performs transformations and generates
    embeddings. CPU backend (scanpy)."""

    transform_funcs = [
        "arcsinh",
        "scale",
        "percentile",
        "zscore",
        "log1p",
    ]

    def __init__(self, adata: AnnData | None) -> None:
        self.adata = adata

        #: Tracks the applied functions and their parameters
        self.applied_funcs = []

    def set_X_to_raw(self):
        """Set .X to .raw.X"""
        self.adata.X = self.adata.raw.X.copy()

    def _warn_if_applied(self, partial_function):
        # Trim inplace
        WARN_MESSAGE = f"WARNING: Already applied {partial_function.func.__name__} to this set of adatas."
        if partial_function.func in self.applied_funcs:
            print(WARN_MESSAGE)

    def _raise_error_if_applied(self, partial_function):
        if partial_function.func in self.applied_funcs:
            raise RuntimeError(
                "Function already applied. Skipping operation..."
            )

    def _cache_applied_function(self, partial_function, tandem_apply):
        if tandem_apply:
            self._warn_if_applied(partial_function)
        else:
            self._raise_error_if_applied(partial_function)
        # Add function, supplied params, and default params
        function_name = partial_function.func.__name__
        function_kwargs = partial_function.keywords
        self.applied_funcs.append(
            {function_name: function_kwargs}
        )  # {function : {'a': val, 'b': val}}}

    def _process(self, adata, attr, partial_function, returns=False):
        if attr is None:
            if returns:
                return partial_function(adata)
            else:
                partial_function(adata)
        else:
            setattr(adata, attr, partial_function(getattr(adata, attr)))
            if returns:
                return adata

    def apply_all(
        self,
        function: Callable,
        attr: Optional[str] = None,
        tandem_apply: Optional[bool] = False,
        **function_kwargs,
    ) -> None:
        """Apply functions to the entire AnnData object, or to a specific
        attribute of the AnnData object.

        Args:
            function: Function to apply.
            attr: Attribute of the AnnData object to apply the function to.
            tandem_apply: If True, allows the same function to be applied
                multiple times.
            **function_kwargs: Additional keyword arguments to pass to the
                function.
        """
        # Freeze applied function
        partial_function = partial(function, **function_kwargs)
        # Save and log function
        self._cache_applied_function(partial_function, tandem_apply)
        self._process(self.adata, attr, partial_function, False)

    # Inject transformation method
    def transform(self):
        """Performs a typical transformation recipe suitable for CODEX/PCF
        intensity-derived expression data on the contained AnnData object.

        1) Arcsinh normalisation, cofactor=150,
        2) Scale columns and rows (0 mean, unit variance)
        3) Z-score X expr. values along columns
        """
        # 1) Arcsinh
        self.arcsinh_transform(cofactor=150)
        # 2) Scale; care for scaling
        self.scale_transform()
        # 3) Z-score
        self.zscore_transform()

    def fill_na(self, fill_val=0.0):
        """Fill NaNs with a given value."""
        self.apply_all(np.nan_to_num, attr="X", nan=fill_val)

    def log1p_transform(self):
        """Log1p transformation"""
        self.apply_all(np.log1p, attr="X")

    def arcsinh_transform(self, cofactor=150):
        """CODEX/fusion transformation: Arcsinh normalisation, cofactor=150"""
        print(
            f"Applying arcsinh transformation with cofactor: {cofactor}",
            flush=True,
        )

        def _arcsinh_transform(X, cofactor):
            return np.arcsinh(X / cofactor)

        self.apply_all(_arcsinh_transform, attr="X", cofactor=cofactor)
        # self.apply_all(lambda x: np.arcsinh(x / cofactor), attr="X"

    def zscore_transform(self):
        """Z-score X expr. values along rows. A given cell's expression across all genes/markers will be scored -1 to 1"""
        self.apply_all(stats.zscore, attr="X", axis=1)

    def scale_transform(self):
        """Scale columns and rows (0 mean, unit variance)"""
        self.apply_all(sc.pp.scale)
        self.fill_na(0.0)  # Fill NaNs with 0s;

    def percentile_transform(self, percentile=95):
        """IMC data normalises to the 95th or 99th percentile. Axis = 0, or per column.
        source: ..."""
        assert percentile > 0  # avoid div by 0

        def _percentile_transform(X, percentile, axis):
            X = np.nan_to_num(X, 0)
            min_val_per_axis = np.min(X, axis=axis)
            percentile_val_per_axis = np.percentile(X, percentile, axis=axis)
            # Below will return divide by 0 runtime error if trying to normalise a column with all 0s,
            # to skip that, we replace the 0s with 1s, and then divide by 1s, which is the same as not dividing.
            denominator = percentile_val_per_axis - min_val_per_axis
            if np.any(denominator == 0):
                denominator[denominator == 0] = 1
            normalised_X = (X - min_val_per_axis) / denominator
            return normalised_X

        self.apply_all(
            _percentile_transform, attr="X", percentile=percentile, axis=0
        )

    def call_transform(self, transform_name: str):
        """Call a transform function by name."""
        if transform_name in self.transform_funcs:
            getattr(self, transform_name + "_transform")()
        else:
            raise ValueError(f"Invalid transform function: {transform_name}")

    def trim_kwargs(self, function_kwargs, function):
        """Trim function_kwargs to only those accepted by function."""
        return {
            k: v
            for k, v in function_kwargs.items()
            if k in function.__code__.co_varnames
        }

    def neighbors(self, **kwargs):
        """Wrapper for `scanpy.pp.neighbors`."""
        kwargs = self.trim_kwargs(kwargs, sc.pp.neighbors)
        sc.pp.neighbors(self.adata, **kwargs)

    def pca(self, **kwargs):
        """Wrapper for `scanpy.tl.pca`."""
        kwargs = self.trim_kwargs(kwargs, sc.tl.pca)
        sc.tl.pca(self.adata, **kwargs)

    def umap(self, **kwargs):
        """Wrapper for `scanpy.tl.umap`."""
        kwargs = self.trim_kwargs(kwargs, sc.tl.umap)
        sc.tl.umap(self.adata, **kwargs)

    def tsne(self, **kwargs):
        """Wrapper for `scanpy.tl.tsne`."""
        kwargs = self.trim_kwargs(kwargs, sc.tl.tsne)
        sc.tl.tsne(self.adata, **kwargs)

    def harmony(self, **kwargs):
        """Wrapper for `scanpy.external.pp.harmony_integrate`."""
        kwargs = self.trim_kwargs(kwargs, sc.external.pp.harmony_integrate)
        sc.external.pp.harmony_integrate(
            self.adata, **kwargs
        )  # key=self.batch_key, max_iter_harmony=40)
        self.harmonised = True

    def get_GPU_version(self) -> AnnDataProcessorGPU:  # type: ignore # noqa F821
        """Returns a copy of the Processor object but with a GPU
        (rapids_singlecell) backend."""
        adata_processor_gpu = AnnDataProcessorGPU(self.adata, skip_init=True)
        adata_processor_gpu.applied_funcs = self.applied_funcs
        adata_processor_gpu.harmonised = self.harmonised
        return adata_processor_gpu


class AnnDataProcessorGPU(AnnDataProcessor):
    """GPU (rapids_singlecell) backend of AnnDataProcessor."""

    def __init__(self, adata: AnnData, skip_init=False):
        super().__init__(adata, skip_init)
        self.check_if_GPU_version_installed()
        self.expr_loc = "CPU"  # Additional attribute to track where the expr. values are stored

    def check_if_GPU_version_installed(self):
        """Check if rapids_singlecell is installed."""
        try:
            import rapids_singlecell  # type: ignore

            self.rsc = rapids_singlecell

        except ImportError as e:
            raise ImportError("rapids_singlecell not installed") from e

    def to_GPU(self):
        """Move stored data structures (.X) to GPU*; similar to PyTorch."""
        if self.expr_loc == "GPU":
            print("X matrices already in GPU.")
        else:
            self.rsc.get.anndata_to_GPU(self.adata)
            self.expr_loc = "GPU"

    def to_CPU(self):
        """Move .X back to CPU."""
        if self.expr_loc == "CPU":
            print("X matrices already in CPU.")
        else:
            self.rsc.get.anndata_to_CPU(self.adata)
            self.expr_loc = "CPU"

    def _check_if_rsc(self, partial_function):
        return partial_function.func.__module__.startswith("rapids_singlecell")

    def check_and_move_to_GPU(self):
        """Moves data to GPU if not already there."""
        if self.expr_loc == "CPU":  # Implicit data memory movement
            self.to_GPU()

    def check_and_move_to_CPU(self):
        """Moves data to CPU if not already there."""
        if self.expr_loc == "GPU":
            self.to_CPU()

    def pca(self, **kwargs):
        """Wrapper for `rapids_singlecell.pp.pca`."""
        kwargs = self.trim_kwargs(kwargs, self.rsc.tl.pca)
        self.check_and_move_to_GPU(self.adata)
        self.rsc.pp.pca(self.adata, **kwargs)

    def neighbors(self, **kwargs):
        """Wrapper for `rapids_singlecell.pp.neighbors`."""
        kwargs = self.trim_kwargs(kwargs, self.rsc.pp.neighbors)
        self.check_and_move_to_GPU(self.adata)
        self.rsc.pp.neighbors(self.adata, **kwargs)

    def umap(self, **kwargs):
        """Wrapper for `rapids_singlecell.tl.umap`."""
        kwargs = self.trim_kwargs(kwargs, self.rsc.tl.umap)
        self.check_and_move_to_GPU(self.adata)
        self.rsc.tl.umap(self.adata, **kwargs)

    def tsne(self, **kwargs):
        """Wrapper for `rapids_singlecell.tl.tsne`."""
        kwargs = self.trim_kwargs(kwargs, self.rsc.tl.tsne)
        self.check_and_move_to_GPU(self.adata)
        self.rsc.tl.tsne(self.adata, **kwargs)

    def harmony(self, **kwargs):
        """Wrapper for `rapids_singlecell.pp.harmony_integrate`."""
        kwargs = self.trim_kwargs(kwargs, self.rsc.pp.harmony_integrate)
        self.rsc.pp.harmony_integrate(self.adata, **kwargs)
        self.harmonised = True

    def get_CPU_version(self):
        """Returns a copy of the Processor object but with a CPU
        (scanpy) backend."""
        adata_processor_cpu = AnnDataProcessor(self.adata, skip_init=True)
        adata_processor_cpu.applied_funcs = self.applied_funcs
        adata_processor_cpu.harmonised = self.harmonised
        return adata_processor_cpu


####################


def split_by_obs(
    adata: AnnData, obs_var: str, selections: Optional[list[str]] = None
):
    """Splits an AnnData by rows, according to an .obs column.
    i.e.) If three unique images in the AnnData, and split by image, then
          produces 3 AnnDatas, each by those images."""
    if selections is None:  # By default, all unique obs_var elements
        selections = adata.obs[obs_var].unique()
    adata_list = []
    for selection in selections:
        adata_list.append(
            adata[adata.obs[obs_var] == selection].copy()
        )  # TODO; try catch selections
    print(f"Got {adata}")
    print(f"Spliting by {obs_var}")
    print("" if selections is None else f"Got selections: {selections}")

    return adata_list, obs_var


class AnnDataCollection:
    """Processes Anndata objects the same way and in parallel."""

    def __init__(
        self, adatas: Union[list[AnnData], AnnData], obs_var: str = None
    ):
        self.adatas = adatas
        self.batch_key = obs_var
        self.merged = None
        self.applied_funcs = []  # Track applied functions;
        self.harmonised = False
        self.determine_mulitple_state()

    def determine_multiple_state(self):
        """Track if collecting only a single or multiple AnnDatas"""
        if len(self.adatas) == 1:
            self.multiple = False
        else:
            self.multiple = True

    def _apply_parallel(self, partial_function, attr):
        if attr is None:
            if partial_function.func.__module__.startswith("scanpy"):
                partial_function = partial(partial_function, copy=True)
            else:  # Assume pm
                partial_function = partial(partial_function, inplace=False)
        adatas = Parallel(n_jobs=len(self.adatas), prefer="processes")(
            delayed(self._process)(adata, attr, partial_function, True)
            for adata in self.adatas
        )
        self.adatas = adatas

    def _apply_normal(self, partial_function, attr):
        for adata in self.adatas:
            self._process(adata, attr, partial_function, False)

    def apply_all(
        self,
        function: Callable,
        attr: Optional[str] = None,
        parallelise: Optional[bool] = False,
        tandem_apply: Optional[
            bool
        ] = False,  # If u can apply the same function twice in a row
        **function_kwargs,
    ):
        """Apply functions to all instance of adata, or adata attrs."""
        # apply the function to the adata inplace, or the attr and set attr.

        # Freeze applied function
        partial_function = partial(function, **function_kwargs)
        # Save and log function
        self._cache_applied_function(partial_function, tandem_apply)

        # GIL; worker processes run in isolated memory spaces,
        if parallelise:  # Then it must return the object, due to above
            self._apply_parallel(partial_function, attr)

        else:  # Below can do inplace memory ops.
            self._apply_normal(partial_function, attr)

    def subset_all(self, function: Callable, **function_kwargs):
        """Same as apply_all, but for functions which change the shape of
        AnnData. Some hard rules:
            - Functions which change columns or .var shape must result in
               the same number of columns across all AnnData objects.
            - Functions must always return the AnnData object
        Functions which change rows across AnnData objects are fine, as we
        combine rows regardless into the final merged dataframe."""

        def _check_columns_post_subset(adata, columns):
            """Checks hard rule 1) above."""

        def _check_rows_post_subset(adata, rows):
            """If a row subset completely removes an AnnData, remove that
            AnnData from the list."""

        def _check_function_returns_adata(adata):
            pass

        # No parallel processing needed here
        self.adatas = [
            function(adata, **function_kwargs) for adata in self.adatas
        ]

    def concatenate_all(self):
        """Ensure non-duplicate obs names."""
        if self.merged is None:
            self.merged = ad.concat(self.adatas)
            self.merged.obs = self.merged.obs.reset_index(drop=True)
            assert self.merged.obs.index.nunique() == self.merged.shape[0]
        return self.merged

    def check_pca(self, compute: bool, save_plot: str = None):
        """Check embeddings if batch/diff adatas vary."""
        if self.merged is None:
            self.concatenate_all()

        if compute and "X_pca" not in self.merged.obsm:
            sc.tl.pca(self.merged)

        if save_plot:
            sc.pl.pca(self.merged, color=self.batch_key, save=save_plot)
        else:
            sc.pl.pca(self.merged, color=self.batch_key)

    def harmonise_adatas(self):
        """Sets the default PCA embedding to the harmonised one."""
        if "X_pca" not in self.merged.obsm:
            raise ValueError("Run pca first.")

        sc.external.pp.harmony_integrate(
            self.merged, key=self.batch_key, max_iter_harmony=40
        )
        self.merged.obsm["X_pca_d"] = self.merged.obsm["X_pca"]
        self.merged.obsm["X_pca"] = self.merged.obsm["X_pca_harmony"]
        self.harmonised = True

    def get_GPU_version(self):
        """Copy func"""
        adata_collection_gpu = AnnDataCollectionGPU(
            self.adatas, self.batch_key
        )
        adata_collection_gpu.applied_funcs = self.applied_funcs
        adata_collection_gpu.merged = self.merged
        adata_collection_gpu.harmonised = self.harmonised
        return adata_collection_gpu


class AnnDataCollectionGPU(AnnDataCollection):
    """GPU backend of AnnDataCollection."""

    def __init__(
        self, adatas: Union[list[AnnData], AnnData], obs_var: str = None
    ):
        super().__init__(adatas, obs_var)
        self.check_if_GPU_version_installed()
        self.expr_loc = "CPU"  # Additional attribute to track where the expr. values are stored

    def check_if_GPU_version_installed(self):
        try:
            import rapids_singlecell  # type: ignore

            self.rsc = rapids_singlecell

        except ImportError as e:
            raise ImportError("rapids_singlecell not installed") from e

    def to_GPU(self, which="all"):
        """Move stored data structures (.X) to GPU*; similar to PyTorch."""
        if self.expr_loc == "GPU":
            print("X matrices already in GPU.")
        else:
            if which == "all":
                for adata in self.adatas:
                    self.rsc.get.anndata_to_GPU(adata)
            elif which == "merged":
                self.rsc.get.anndata_to_GPU(self.merged)
            else:
                raise ValueError("Invalid data var.")
            self.expr_loc = "GPU"

    def to_CPU(self, which="all"):
        """Move .X back to CPU."""
        if self.expr_loc == "CPU":
            print("X matrices already in CPU.")
        else:
            if which == "all":
                for adata in self.adatas:
                    self.rsc.get.anndata_to_CPU(adata)
            elif which == "merged":
                self.rsc.get.anndata_to_CPU(self.merged)
            else:
                raise ValueError("Invalid data var.")
            self.expr_loc = "CPU"

    def _check_if_rsc(self, partial_function):
        return partial_function.func.__module__.startswith("rapids_singlecell")

    def _check_and_move_to_GPU(self, which):
        if self.expr_loc == "CPU":  # Implicit data memory movement
            self.to_GPU(which)

    def _check_and_move_to_CPU(self, which):
        if self.expr_loc == "GPU":
            self.to_CPU(which)

    # Overload
    def _apply_parallel(self, partial_function, attr):
        """Overloaded, checks if function is RSC/gpu, then raises error.
        Here, simply raising NotImplementedError, since our operations
        with parallel GPUs is likely overkill."""

        if self._check_if_rsc(partial_function):
            raise NotImplementedError(
                "Parallel GPU implementations unsupported."
            )

        if attr is None:
            if partial_function.func.__module__.startswith("scanpy"):
                partial_function = partial(partial_function, copy=True)
            else:  # Assume pm
                partial_function = partial(partial_function, inplace=False)
        adatas = Parallel(n_jobs=len(self.adatas), prefer="processes")(
            delayed(self._process)(adata, attr, partial_function, True)
            for adata in self.adatas
        )
        self.adatas = adatas

    # Overload
    def _apply_normal(self, partial_function, attr):
        """Overloaded, checks if function is RSC/gpu, then implicitly checks
        and moves data to GPU for computations."""

        if self._check_if_rsc(partial_function):
            self._check_and_move_to_GPU("all")

        for adata in self.adatas:
            self._process(adata, attr, partial_function, False)

    def harmonise_adatas(self):
        """Sets the default PCA embedding to the harmonised one."""
        if "X_pca" not in self.merged.obsm:
            raise ValueError("Run pca first.")

        # CPU -> GPU
        self._check_and_move_to_GPU("merged")

        self.rsc.pp.harmony_integrate(
            self.merged, key=self.batch_key, max_iter_harmony=40
        )

        self.merged.obsm["X_pca_d"] = self.merged.obsm["X_pca"]
        self.merged.obsm["X_pca"] = self.merged.obsm["X_pca_harmony"]
        self.harmonised = True

        # GPU -> CPU
        self._check_and_move_to_CPU("merged")


def pcf_factory(adatas: list[AnnData], obs_var: str = None, use_gpu=False):
    """Factory function to return the appropriate AnndataCollection object."""
    if use_gpu:
        print(f"Using GPU, supplying: {adatas}")
        pcf_collection = AnnDataCollectionGPU(adatas, obs_var)
    else:
        print(f"Using CPU, supplying: {adatas}")
        pcf_collection = AnnDataCollection(adatas, obs_var)

    return pcf_collection

""".pp module"""

import importlib
from collections.abc import Callable
from functools import wraps
from typing import Literal, Optional

import loguru
import numpy as np
import pandas as pd
import scipy.stats as stats
from anndata import AnnData

_current_backend = {"module": "scanpy"}
sc_backend = importlib.import_module(_current_backend["module"])


def set_backend(backend: Literal["cpu", "gpu"]) -> None:
    """Set the backend to use for processing."""
    global sc_backend
    if backend == "cpu":
        _current_backend["module"] = "scanpy"
        loguru.logger.info("Setting backend to CPU with scanpy")
    elif backend == "gpu":
        try:
            import rapids_singlecell  # noqa F401

            _current_backend["module"] = "rapids_singlecell"
            loguru.logger.info("Setting backend to GPU with rapids_singlecell")
        except ImportError as e:
            raise ImportError("rapids_singlecell not installed") from e
    else:
        raise ValueError("Invalid backend. Must be 'cpu' or 'gpu'")

    sc_backend = importlib.import_module(_current_backend["module"])


def with_current_backend(function: Callable) -> Callable:
    """Decorator to dynamically use current backend for scanpy-type functions.
    Also trims keyword arguments to only those accepted by the function.

    If GPU backend is set, then function handles moving data to GPU memory.
    After running the function, it always returns it back to CPU memory.

    Args:
        function: Scanpy or rapids_singlecell function to wrap.

    Returns:
        Wrapped function.
    """

    @wraps(function)
    def wrapper(adata, **kwargs):
        backend = _current_backend["module"]
        if backend == "rapids_singlecell":
            sc_backend.get.anndata_to_GPU(adata)

        function_kwargs = trim_kwargs(kwargs, function)
        result = function(adata, **function_kwargs)

        if backend == "rapids_singlecell":
            if result is None:
                sc_backend.get.anndata_to_CPU(adata)
            else:
                sc_backend.get.anndata_to_CPU(result)

        return result

    return wrapper


def trim_kwargs(function_kwargs: dict, function: Callable) -> dict:
    """Trim function_kwargs to only those accepted by function.

    Args:
        function_kwargs: Keyword arguments to trim.
        function: Function to trim keyword arguments for.

    Returns:
        Trimmed keyword arguments.
    """
    return {
        k: v
        for k, v in function_kwargs.items()
        if k in function.__code__.co_varnames
    }


def filter_by_obs_count(
    adata: AnnData,
    obs_col: str,
    min_value: float | None = None,
    max_value: float | None = None,
    copy: bool = True,
) -> AnnData:
    """Filters cells which belong to a category in `AnnData.obs` with a cell
    count less than a `min_value` and/or more than a `max_value`. If layer is
    None, does this on .X.

    Args:
        adata: Anndata object.
        obs_col: `AnnData.obs` column to filter by.
        min_value: Minimum value to filter by.
        max_value: Maximum value to filter by.
        copy: Return a copy instead of writing inplace.

    Returns:
        Filtered AnnData object. If `copy` is False, modifies the AnnData object
        in place and returns None.
    """
    MIN_DEFAULT = 10
    MAX_DEFAULT = 1000000

    if copy:
        adata = adata.copy()

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

    adata = adata[adata.obs[obs_col].isin(cells_by_obs.index)]

    if copy:
        return adata


def filter_by_obs_value(
    adata: AnnData,
    obs_col: str,
    min_value: float | None = None,
    max_value: float | None = None,
    copy: bool = True,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.obs` column less
    than a `min_value` and/or more than a `max_value`.

    Args:
        adata: Anndata object.
        obs_col: `AnnData.obs` column to filter by.
        min_value: Minimum value to filter by.
        max_value: Maximum value to filter by.
        copy: Return a copy instead of writing inplace.

    Returns:
        Filtered AnnData object. If `copy` is False, modifies the AnnData object
        in place and returns None.
    """

    MIN_DEFAULT = 10
    MAX_DEFAULT = 100000  # 255 8 bit, 1 pixel below

    if copy:
        adata = adata.copy()

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

    if copy:
        return adata


def filter_by_obs_quantile(
    adata: AnnData,
    obs_col: str,
    min_quantile: float | None = None,
    max_quantile: float | None = None,
    copy: bool = True,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.obs` column less
    than the `min_quantile` and/or more than the `max_quantile`.

    Args:
        adata: Anndata object.
        obs_col: `AnnData.obs` column to filter by.
        min_quantile: Minimum quantile to filter by.
        max_quantile: Maximum quantile to filter by.
        copy: Return a copy instead of writing inplace.

    Returns:
        Filtered AnnData object. If `copy` is False, modifies the AnnData object
        in place and returns None.
    """
    MIN_DEFAULT = 0.1
    MAX_DEFAULT = 0.95

    if copy:
        adata = adata.copy()

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

    if copy:
        return adata


def filter_by_var_value(
    adata: AnnData,
    var: str,
    min_value: float | None = None,
    max_value: float | None = None,
    layer: str | None = None,
    copy: bool = True,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.var` column less
    than a `min_value` and/or more than a `max_value`.

    Args:
        adata: Anndata object.
        var: `AnnData.var` column to filter by.
        min_value: Minimum value to filter by.
        max_value: Maximum value to filter by.
        layer: Expression layer to filter.
        copy: Return a copy instead of writing inplace.


    Returns:
        Filtered AnnData object. If `copy` is False, modifies the AnnData object
        in place and returns None.
    """
    MIN_DEFAULT = 1  # 0 pixels above
    MAX_DEFAULT = 254  # 255 8 bit, 1 pixel below

    if copy:
        adata = adata.copy()

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

    if copy:
        return adata


def filter_by_var_quantile(
    adata: AnnData,
    var: str,
    min_quantile: float | None = None,
    max_quantile: float | None = None,
    layer: str | None = None,
    copy: bool = True,
) -> AnnData:
    """Filters cells which have a value of a given `AnnData.var` column less
    than the `min_quantile` and/or more than the `max_quantile`.

    Args:
        adata: Anndata object.
        var: `AnnData.var` column to filter by.
        min_quantile: Minimum quantile to filter by.
        max_quantile: Maximum quantile to filter by.
        layer: Expression layer to filter.
        copy: Return a copy instead of writing inplace.

    Returns:
        Filtered AnnData object. If `copy` is False, modifies the AnnData object
        in place and returns None.
    """
    MIN_DEFAULT = 0.05
    MAX_DEFAULT = 0.99

    if copy:
        adata = adata.copy()

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

    if copy:
        return adata


def fill_na(
    adata: AnnData,
    fill_value: float = 0.0,
    layer: str | None = None,
    copy: bool = True,
) -> AnnData:
    """Fill NaNs in a given layer or .X with a given value.

    Args:
        adata: Anndata object.
        fill_value: Value to fill NaNs with.
        layer: Expression layer to fill NaNs.
        copy: Return a copy instead of writing inplace.

    Returns:
        AnnData object with NaNs filled. If `copy` is False, modifies the
        AnnData object in place and returns None.
    """
    if copy:
        adata = adata.copy()

    if layer is None:
        adata.X = np.nan_to_num(adata.X, fill_value)
    else:
        adata.layers[layer] = np.nan_to_num(adata.layers[layer], fill_value)

    if copy:
        return adata


@with_current_backend
def log1p(adata: AnnData, copy: bool = True, **kwargs) -> AnnData:
    """Apply log1p transformation (natural log transform with pseudocount) to a
    given layer or .X. Wraps `sc.pp.log1p` or `rsc.pp.log1p`.

    Args:
        adata: Anndata object.
        copy: Return a copy instead of writing inplace.
        **kwargs: Additional keyword arguments to pass to `pp.log1p`.

    Returns:
        New AnnData object with log1p transformation applied. If `copy` is
            False, modifies the AnnData object in place and returns None.
    """
    return sc_backend.pp.log1p(adata, copy=copy, **kwargs)


def arcsinh(
    adata: AnnData,
    cofactor: float = 150,
    layer: str | None = None,
    copy: bool = True,
) -> AnnData:
    """Apply arcsinh transformation with a given cofactor to a given layer or
    .X.

    Args:
        adata: Anndata object.
        cofactor: Cofactor for arcsinh transformation. Defaults to 150.
        layer: Expression layer to apply arcsinh transformation. If `None`, `X`
            is transformed.
        copy: Return a copy instead of writing inplace.

    Returns:
        AnnData object with arcsinh transformation applied. If `copy` is False,
        modifies the AnnData object in place and returns None.
    """
    if copy:
        adata = adata.copy()

    if layer is None:
        adata.X = np.arcsinh(adata.X / cofactor)
    else:
        adata.layers[layer] = np.arcsinh(adata.layers[layer] / cofactor)

    if copy:
        return adata


def zscore(
    adata: AnnData,
    layer: str | None = None,
    copy: bool = True,
) -> AnnData:
    """Apply z-score transformation along the rows of a given layer or .X. Each
    cell's expression across vars will be scored -1 to 1.

    i.e. A relative 'rank' of vars within each cell.

    Args:
        adata: Anndata object.
        layer: Expression layer to apply z-score transformation. If `None`, `X`
            is transformed.
        copy: Return a copy instead of writing inplace.

    Returns:
        AnnData object with row-wise z-score transformation applied. If `copy`
        is False, modifies the AnnData object in place and returns None.
    """
    if copy:
        adata = adata.copy()

    if layer is None:
        adata.X = stats.zscore(adata.X, axis=1)
    else:
        adata.layers[layer] = stats.zscore(adata.layers[layer], axis=1)

    if copy:
        return adata


@with_current_backend
def scale(
    adata: AnnData, layer: str | None = None, copy: bool = True, **kwargs
) -> AnnData:
    """Scale columns and rows to have 0 mean and unit variance in a given layer
    or .X. Wraps `scanpy.pp.scale` or `rsc.pp.scale`.

    Args:
        adata: Anndata object.
        layer: Expression layer to apply scale transformation. If `None`, `X` is
            transformed.
        copy: Return a copy instead of writing inplace.
        **kwargs: Additional keyword arguments to pass to `pp.scale`.

    Returns:
        AnnData object with scale transformation applied. If `copy` is False,
        modifies the AnnData object in place and returns None
    """
    return sc_backend.pp.scale(adata, layer=layer, copy=copy, **kwargs)


def percentile(
    adata: AnnData,
    percentile: float = 95,
    layer: str | None = None,
    copy: bool = True,
) -> AnnData:
    """Normalise data to the 95th or 99th percentile. Axis = 0, or per column.
    Wraps `np.percentile`.

    Args:
        adata: Anndata object.
        percentile: Percentile to normalise to. Defaults to 95.
        layer (optional): Expression layer to apply percentile transformation.
            If `None`, `X` is transformed.
        copy: Return a copy instead of writing inplace.

    Returns:
        AnnData object with percentile transformation applied. If `copy` is
        False, modifies the AnnData object in place and returns None.
    """

    def _percentile_transform(X, percentile, axis):
        X = np.nan_to_num(X, 0)
        min_val_per_axis = np.min(X, axis=axis)
        percentile_val_per_axis = np.percentile(X, percentile, axis=axis)
        # Below will return divide by 0 runtime error if trying to normalise
        # a column with all 0s,
        # to skip that, we replace the 0s with 1s, and then divide by 1s, which
        # is the same as not dividing.
        denominator = percentile_val_per_axis - min_val_per_axis
        if np.any(denominator == 0):
            denominator[denominator == 0] = 1
        normalised_X = (X - min_val_per_axis) / denominator
        return normalised_X

    if copy:
        adata = adata.copy()

    if layer is None:
        adata.X = _percentile_transform(adata.X, percentile, axis=0)
    else:
        adata.layers[layer] = _percentile_transform(
            adata.layers[layer], percentile, axis=0
        )

    if copy:
        return adata


@with_current_backend
def neighbors(
    adata: AnnData,
    copy: bool = True,
    **kwargs,
) -> AnnData:
    """Compute a neighborhood graph in a given expression or embedding space.
    Wrapper for `scanpy.pp.neighbors` or `rsc.pp.neighbors`.

    Args:
        adata: Anndata object.
        copy: Return a copy of the modified AnnData object.
        **kwargs: Additional keyword arguments to pass to `pp.neighbors`.

    Returns:
        AnnData object with neighbors computed. If `copy` is False, modifies the
        AnnData object in place and returns None.
    """
    return sc_backend.pp.neighbors(adata, copy=copy, **kwargs)


# TODO: Legacy/Deprecate
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

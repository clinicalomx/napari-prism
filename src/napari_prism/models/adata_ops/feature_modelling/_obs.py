"""Classes for generating sample-level AnnData objects."""

from typing import Literal

import pandas as pd


class ObsAggregator:
    AGGREGATION_ERROR = (
        "`aggregation_function` not supported or none provided. \n"
        "Valid aggregation functions: \n\t"
        "sum, max, min, first, last, mean, median."
    )
    CATEGORICAL_HANDLES = ["category_counts", "category_proportions"]
    NUMERICAL_HANDLES = ["summarise", "bin", "widen"]
    """
    Obs Helper for retrieving metadata with common indexes relative to a
    specified parent column.

    Useful for generating sample-level metrics from cell-level metadata.
    """

    def __init__(self, adata, base_column):
        self.adata = adata
        self.base_column = base_column

        #: Key Relations of base_columns to every other column in adata.obs
        self.parallel_keys = None
        self.super_keys = None
        self.categorical_keys = None
        self.numerical_keys = None
        self._set_groupby_df(base_column)
        self._set_column_relations()
        self._log_column_dtypes()

    @staticmethod
    def get_groupby_df(adata, base_column):
        return adata.obs.groupby(base_column, observed=False)

    @staticmethod
    def get_cardinality_df(adata, base_column):
        groupby_df = ObsAggregator.get_groupby_df(adata, base_column)
        return groupby_df.nunique()

    @staticmethod
    def get_parallel_keys(adata, base_column):
        """Get the keys which have a 1 or N : 1 relation with base_column."""
        cardinality_df = ObsAggregator.get_cardinality_df(adata, base_column)
        oto = (
            cardinality_df.sum() <= cardinality_df.shape[0]
        )  # If less then, then NaNs
        return cardinality_df.columns[oto]

    @staticmethod
    def get_true_parallel_keys(adata, base_column):
        """Get the keys which have a 1 : 1 relation with base_column."""
        return adata.obs.columns[
            adata.obs.nunique() == adata.obs[base_column].nunique()
        ]

    @staticmethod
    def get_duplicated_keys(adata, base_column):
        """Get the keys which have a N : 1 relation with base_column.
        i.e. The values in these keys are repeated across multiple
        base_column instances."""
        parallel_keys = ObsAggregator.get_parallel_keys(adata, base_column)
        return adata.obs[parallel_keys].columns[
            adata.obs[parallel_keys].nunique()
            < adata.obs[base_column].nunique()
        ]

    @staticmethod
    def get_super_keys(adata, base_column):
        """Get the keys which have a 1 : N relation with base_column."""
        cardinality_df = ObsAggregator.get_cardinality_df(adata, base_column)
        oto = cardinality_df.sum() > cardinality_df.shape[0]
        return cardinality_df.columns[oto]

    def _set_groupby_df(self, base_column):
        self.groupby_df = ObsAggregator.get_groupby_df(self.adata, base_column)

    def _set_column_relations(self):
        """Based on cardinality_df, will get keys which have a 1:1 relation."""
        self.parallel_keys = ObsAggregator.get_parallel_keys(
            self.adata, self.base_column
        )
        self.super_keys = ObsAggregator.get_super_keys(
            self.adata, self.base_column
        )

    def _log_column_dtypes(self):
        """Log the data types of each key."""
        df = self.adata.obs
        # categorical_dtypes = df.select_dtypes(exclude="number").columns
        numerical_dtypes = df.select_dtypes(include="number").columns
        # If the key is numerical AND a super key then its a true numeric which needs aggregation
        self.numerical_keys = pd.Index(
            [x for x in numerical_dtypes if x in self.super_keys]
        )

        # If the key is numerical but a parallel key then it can be treated like a categorical parallel key
        categorical_numerics = pd.Index(
            [x for x in numerical_dtypes if x in self.parallel_keys]
        )
        self.categorical_keys = df.select_dtypes(exclude="number").columns
        self.categorical_keys = self.categorical_keys.append(
            categorical_numerics
        )

    def get_metadata_df(
        self,
        column,
        *,
        additional_groupby=None,
        skey_handle=None,
        aggregation_function=None,
        bins=None,
    ):
        if additional_groupby is None:
            groupby_obj = self.groupby_df
        else:
            if isinstance(additional_groupby, str):
                additional_groupby = [additional_groupby]
            groupby_obj = ObsAggregator.get_groupby_df(
                self.adata, [self.base_column] + additional_groupby
            )

        def _get_parallel_key(groupby_obj, column):
            groupby_obj = groupby_obj[column]
            assert all(groupby_obj.nunique() <= 1)
            return groupby_obj.first()

        def _get_super_key(
            groupby_obj, column, skey_handle, aggregation_function, bins
        ):
            # Directive A) Categorical;
            def _get_super_key_categorical(groupby_obj, column, skey_handle):
                # Directive 1: Rows = base, Columns = each category in column, Values = Counts of that category per base.
                if skey_handle in self.CATEGORICAL_HANDLES:
                    vals = groupby_obj[column].value_counts().unstack(column)
                    if skey_handle == "category_proportions":
                        vals = vals.div(vals.sum(axis=1), axis=0)
                    return vals
                else:
                    raise ValueError(
                        "Unsupported skey handle for categorical superkey column"
                    )

            # Directive B) Numerical
            def _get_super_key_numerical(
                groupby_obj, column, skey_handle, aggregation_function, bins
            ):
                # Sub-Directive B1) Numerical -> Categorical; Binning Agg -> Parsed to Directive A
                def _bin_numerics(
                    groupby_obj, column, bins
                ):  # define bins as a list of nums defining boundaries; i.e. [-np.inf, -50, 0, 50, np.inf]
                    assert bins is not None

                    def _bin_and_count(groupby_obj, column, bins):
                        # Apply binning
                        binned = pd.cut(groupby_obj[column], bins=bins)
                        counts = binned.value_counts().reindex(
                            pd.IntervalIndex.from_breaks(bins, closed="right")
                        )
                        return counts

                    output_df = groupby_obj.apply(
                        _bin_and_count, column=column, bins=bins
                    )
                    output_df.columns.name = f"{column}_counts"
                    return output_df

                # Sub-Directive B2) Numerical -> Summary per base. (i.e. mean dist {column} per unique_core {base})
                def _summarise_numerics(
                    groupby_obj, column, aggregation_function
                ):
                    def _get_aggregation_function(aggregation_function):
                        # Parse common aggregation functions which are str to pd.core.GroupBy callables
                        match aggregation_function:  # Pass
                            case "sum":
                                return pd.core.groupby.DataFrameGroupBy.sum
                            case "max":
                                return pd.core.groupby.DataFrameGroupBy.max
                            case "min":
                                return pd.core.groupby.DataFrameGroupBy.min
                            case "first":
                                return pd.core.groupby.DataFrameGroupBy.first
                            case "last":
                                return pd.core.groupby.DataFrameGroupBy.last
                            case "mean":
                                return pd.core.groupby.DataFrameGroupBy.mean
                            case "median":
                                return pd.core.groupby.DataFrameGroupBy.median
                            case _:
                                raise ValueError(self.AGGREGATION_ERROR)

                    agg_func = _get_aggregation_function(aggregation_function)
                    output_df = agg_func(groupby_obj[column])
                    # Rename column to specify the aggregation performed
                    output_df.name = f"{aggregation_function}_{column}"
                    return output_df

                # Sub-Directive B3) Numerical Widened -> Restricted to annotation boxplots/scatterplots etc.
                def _widen_numerics(groupby_obj, column):
                    grouped = groupby_obj[column].apply(list)
                    return pd.DataFrame(grouped.tolist(), index=grouped.index)

                # Handle numerical sub-directives
                if skey_handle not in self.NUMERICAL_HANDLES:
                    raise ValueError(
                        "Unsupported skey handle for numerical superkey column"
                    )

                if skey_handle == "summarise":
                    return _summarise_numerics(
                        groupby_obj, column, aggregation_function
                    )
                elif skey_handle == "bin":
                    return _bin_numerics(groupby_obj, column, bins)
                elif skey_handle == "widen":
                    return _widen_numerics(groupby_obj, column)
                else:
                    raise ValueError(
                        "Invalid skey_handling method for numerics."
                    )

            ## Apply appropriate directives
            if isinstance(column, list):
                if all(c for c in column if c in self.categorical_keys):
                    return _get_super_key_categorical(
                        groupby_obj, column, skey_handle
                    )
                else:  # theres a numeric;
                    raise NotImplementedError(
                        "Mixed key cardinalities/dtypes not implemneted yet"
                    )
            else:
                if column in self.categorical_keys:
                    return _get_super_key_categorical(
                        groupby_obj, column, skey_handle
                    )
                else:
                    return _get_super_key_numerical(
                        groupby_obj,
                        column,
                        skey_handle,
                        aggregation_function,
                        bins,
                    )

        # Parallel Keys
        if isinstance(column, list):
            # Assert for now that both at parallel keys
            if all(c for c in column if c in self.parallel_keys):
                result = _get_super_key(
                    groupby_obj,
                    column,
                    skey_handle,
                    aggregation_function,
                    bins,
                )  # INFS theory; if multiple primary keys-> it follows to become a superkey;
        else:
            if column in self.parallel_keys:
                result = _get_parallel_key(groupby_obj, column)
            else:
                result = _get_super_key(
                    groupby_obj,
                    column,
                    skey_handle,
                    aggregation_function,
                    bins,
                )

            if isinstance(result, pd.Series):
                result = pd.DataFrame(result)

        # Then if additional_groupby is given,
        # then we need to unstack the additional_groupby to return to sample
        # based indexing
        if additional_groupby is not None:
            result = result.unstack(level=additional_groupby)
        return result

    def get_metadata_column(self, metadata_df):
        # Mainly for getting the appropriate column metadata from a given metadata_df, with some aggregation usually;
        return pd.DataFrame(
            metadata_df.columns.to_list(),
            columns=[x + "_col" for x in metadata_df.columns.names],
            index=metadata_df.columns,
        )

    # More user-friendly functions
    def get_category_counts(
        self, categorical_column: str | list[str]
    ) -> pd.DataFrame:
        """
        Counts the number of cells of a given category or categories for
        each sample (`self.base_column`).

        Args:
            categorical_column: .obs columns to to compute proportions for.

        Returns:
            pd.DataFrame with the count data where each instance of
            `self.base_column` as indexing rows, and columns as the category or
            categories in `categorical_column`. If multiple categories were
            given, then the columns are MultiIndexed.
        """
        return self.get_metadata_df(
            categorical_column, skey_handle="category_counts"
        )

    def get_category_proportions(
        self, categorical_column: str | list[str], normalisation_column=None
    ) -> pd.DataFrame:
        """
        Computes the proportion of cells of a given category or categories for
        each sample (`self.base_column`).

        Args:
            categorical_column: .obs columns to to compute proportions for.
            normalisation_column: .obs column to normalise by. Defaults to None.
                If `normalisation_column` is given, then it normalises by the
                total number of cells in each category of that
                `normalisation_column` rather the total number of cells.

        Returns:
            pd.DataFrame with the proportions data where each instance of
            `self.base_column` as indexing rows, and columns as the category or
            categories in `categorical_column`. If multiple categories were
            given, then the columns are MultiIndexed.
        """
        df = self.get_metadata_df(
            categorical_column, skey_handle="category_proportions"
        )
        if normalisation_column is not None and isinstance(
            categorical_column, list
        ):
            column_indexer = df.columns.names.index(normalisation_column)
            indexer_totals = df.T.groupby(level=column_indexer).sum()
            df = df.div(indexer_totals, level=column_indexer, axis=1)

        return df

    def get_numerical_summarised(
        self,
        numerical_column: str,
        aggregation_function: Literal[
            "min", "max", "sum", "mean", "median", "first", "last"
        ],
        categorical_column: str | list[str] = None,
    ) -> pd.DataFrame:
        """
        Summarises the values of a numerical column for each sample
        (`self.base_column`).

        Args:
            numerical_column: .obs column to summarise.
            aggregation_function: Aggregation function to apply to the
                numerical column. One of ["min", "max", "sum", "mean",
                "median", "first", "last"].
            categorical_column: .obs column(s) to further group by. Defaults to
                None.

        Returns:
            pd.DataFrame with the summarised data where each instance of
            `self.base_column` as indexing rows, and column as the
            summarised numerical column, relabelled as
            "{aggregation_function}_{numerical_column}".
        """

        return self.get_metadata_df(
            numerical_column,
            skey_handle="summarise",
            aggregation_function=aggregation_function,
            additional_groupby=categorical_column,
        )

    def get_numerical_binned(
        self,
        numerical_column: str,
        bins: list[int],
        categorical_column: str | list[str] = None,
        normalise: bool = False,
        normalisation_column: str = None,
    ) -> pd.DataFrame:
        """
        Counts the number of cells belonging to each bin of a numerical variable
        for each sample (`self.base_column`).

        Args:
            numerical_column: .obs column to bin.
            bins: List of numerical values defining the bin boundaries. Must be
                monotonic and have at least two elements.
            categorical_column: .obs column(s) to further group by. Defaults to
                None.
            normalise: Whether to normalise the counts by the total number of
                cells in each sample. Defaults to False.
            normalisation_column: .obs column to normalise by. Defaults to None.
                If `normalisation_column` is given, then it normalises by the
                total number of cells in each category of that
                `normalisation_column`. normalise must be True for this to work.

        Returns:
            pd.DataFrame with the count data where each instance of
            `self.base_column` as indexing rows, and columns as the bin
            categories. The bin categories are defined by the bin boundaries
            with the last bin being open-ended.
        """
        df = self.get_metadata_df(
            numerical_column,
            skey_handle="bin",
            bins=bins,
            additional_groupby=categorical_column,
        )
        if normalise:
            if normalisation_column is None:
                df = df.div(df.sum(axis=1), axis=0)
            else:
                column_indexer = df.columns.names.index(normalisation_column)
                indexer_totals = df.T.groupby(level=column_indexer).sum()
                df = df.div(indexer_totals.T, level=column_indexer, axis=1)
        return df

    def get_numerical_widened(self, numerical_column) -> pd.DataFrame:
        """
        Gets every numerical observation for each sample (`self.base_column`).

        Args:
            numerical_column: .obs column to widen.

        Returns:
            pd.DataFrame with the numerical data where each instance of
            `self.base_column` as indexing rows, and columns as every numerical
            observation for each sample. Columns denote the nth observation
            for each sample. Produces many NaNs.
        """
        return self.get_metadata_df(numerical_column, skey_handle="widen")

    def get_numerical_binned_and_aggregated(
        self,
        numerical_column_to_bin: str,
        numerical_column_to_aggregate: str,
        bins: list[int],
        aggregation_function: Literal[
            "min", "max", "sum", "mean", "median", "first", "last"
        ],
        categorical_column: str | list[str] = None,
    ):
        numerical_as_category_name = f"binned_{numerical_column_to_bin}"
        store_adata_obs = self.adata.obs.copy()
        bin_cells = pd.cut(store_adata_obs[numerical_column_to_bin], bins=bins)
        bin_cells.name = numerical_as_category_name
        merged = store_adata_obs.merge(
            bin_cells, left_index=True, right_index=True
        )
        self.adata.obs = merged

        if categorical_column is not None:
            if isinstance(categorical_column, str):
                categorical_column = [categorical_column]
            cats = categorical_column + [numerical_as_category_name]
        else:
            cats = numerical_as_category_name

        result = self.get_numerical_summarised(
            numerical_column_to_aggregate,
            aggregation_function=aggregation_function,
            categorical_column=cats,
        )
        self.adata.obs = store_adata_obs

        return result


class ObsAnnotator:
    """
    Obs Annotator for labelling cell metadata with from various annotation
    sources (dictionaries, spatial queries, etc).
    """

    def spatial_annotation(self):
        """https://spatialdata.scverse.org/en/stable/tutorials/notebooks/
        notebooks/examples/napari_rois.html
        """

    def annotate_column(self, column_name, mapping, condition, subset):
        pass

    def append_column(self, column_name, mapping, condition, subset):
        pass

    def remove_column(self, column_name):
        pass

    def annotate_column_by_rules(self, column_name, rules):
        """Using a rules based dictionary, annotate column labels
        based on the value(s) of two or more columns."""
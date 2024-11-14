import pandas as pd


class ObsHelper:
    AGGREGATION_ERROR = (
        "`aggregation_function` not supported or none provided. \n"
        "Valid aggregation functions: \n\t"
        "sum, max, min, first, last, mean, median."
    )
    """
    Obs Helper for retrieving metadata with common indexes relative to a
    specified parent column.
    """

    def __init__(self, adata, base_column):
        self.adata = adata
        self.base_column = base_column
        self._set_groupby_df(base_column)
        self._set_column_relations()
        self._log_column_dtypes()

    @staticmethod
    def get_groupby_df(adata, base_column):
        return adata.obs.groupby(base_column, observed=False)

    @staticmethod
    def get_cardinality_df(adata, base_column):
        groupby_df = ObsHelper.get_groupby_df(adata, base_column)
        return groupby_df.nunique()

    @staticmethod
    def get_parallel_keys(adata, base_column):
        """Get the keys which have a 1 or N : 1 relation with base_column."""
        cardinality_df = ObsHelper.get_cardinality_df(adata, base_column)
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
        parallel_keys = ObsHelper.get_parallel_keys(adata, base_column)
        return adata.obs[parallel_keys].columns[
            adata.obs[parallel_keys].nunique()
            < adata.obs[base_column].nunique()
        ]

    @staticmethod
    def get_super_keys(adata, base_column):
        """Get the keys which have a 1 : N relation with base_column."""
        cardinality_df = ObsHelper.get_cardinality_df(adata, base_column)
        oto = cardinality_df.sum() > cardinality_df.shape[0]
        return cardinality_df.columns[oto]

    def _set_groupby_df(self, base_column):
        self.groupby_df = ObsHelper.get_groupby_df(self.adata, base_column)

    def _set_column_relations(self):
        """Based on cardinality_df, will get keys which have a 1:1 relation."""
        self.parallel_keys = ObsHelper.get_parallel_keys(
            self.adata, self.base_column
        )
        self.super_keys = ObsHelper.get_super_keys(
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
        self, column, *, skey_handle=None, aggregation_function=None, bins=None
    ):
        groupby_obj = self.groupby_df

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
                if skey_handle in ["category_counts", "category_proportions"]:
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

                    return groupby_obj.apply(
                        _bin_and_count, column=column, bins=bins
                    )

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
                    return agg_func(groupby_obj[column])

                # Sub-Directive B3) Numerical Widened -> Restricted to annotation boxplots/scatterplots etc.
                def _widen_numerics(groupby_obj, column):
                    grouped = groupby_obj[column].apply(list)
                    return pd.DataFrame(grouped.tolist(), index=grouped.index)

                # Handle numerical sub-directives
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

        return result

    def get_metadata_column(self, metadata_df):
        # Mainly for getting the appropriate column metadata from a given metadata_df, with some aggregation usually;
        return pd.DataFrame(
            metadata_df.columns.to_list(),
            columns=[x + "_col" for x in metadata_df.columns.names],
            index=metadata_df.columns,
        )


class ObsAnnotator:
    def annotate_column(self, column_name, mapping, condition, subset):
        pass

    def append_column(self, column_name, mapping, condition, subset):
        pass

    def remove_column(self, column_name):
        pass

    def annotate_column_by_rules(self, column_name, rules):
        """Using a rules based dictionary, annotate column labels
        based on the value(s) of two or more columns."""

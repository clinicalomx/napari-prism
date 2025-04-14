from enum import Enum

import napari
import numpy as np
import pandas as pd
from anndata import AnnData
from magicgui.widgets import ComboBox, Container, Label, Table, create_widget
from napari.utils.events import EmitterGroup
from spatialdata import SpatialData

from napari_prism.models.adata_ops.feature_modelling._obs import ObsAggregator
from napari_prism.widgets.adata_ops._base_widgets import AnnDataOperatorWidget

NULL_CHOICE = "-----"


class ObsAggregatorWidget(AnnDataOperatorWidget):
    """Interface for using the ObsAggregator class."""

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        adata: AnnData,
        sdata: SpatialData | None = None,
    ) -> None:
        self.latest_result = None  # NOTE: temporary;
        self.model = None
        super().__init__(viewer, adata=adata, sdata=sdata)
        #: Events for when an anndata object is changed
        self.events = EmitterGroup(
            source=self,
            sdata_changed=None,
        )

    def create_parameter_widgets(self) -> None:
        """Create widgets for exposing the functions of the ObsAggregator class."""
        AGGREGATION_FUNCTIONS = [
            "category_counts",
            "category_proportions",
            "numerical_summarised",
            "numerical_binned",
            "numerical_widened",
            "numerical_binned_and_aggregated",
        ]

        self.sample_key = ComboBox(
            name="SampleKey",
            choices=super().get_categorical_obs_keys,
            label="Sample Level Key",
            nullable=True,
        )
        self.sample_key.changed.connect(self.update_local_model)
        self.sample_key.changed.connect(self.reset_choices)

        self.aggregation_functions_selection = ComboBox(
            value=None,
            name="Aggregation function",
            choices=AGGREGATION_FUNCTIONS,
            nullable=True,
        )
        self.aggregation_functions_selection.scrollable = True
        self.aggregation_functions_selection.changed.connect(
            self.local_create_parameter_widgets
        )

        self.aggregation_functions_container = Container()

        self.extend(
            [
                self.sample_key,
                self.aggregation_functions_selection,
                self.aggregation_functions_container,
            ]
        )

    def clear_local_layout(self) -> None:
        # layout = self.aggregation_functions_container.native.layout()
        # for _ in range(layout.count() - 1):
        #     layout.itemAt(1).widget().setParent(None)
        for x in list(self.aggregation_functions_container):
            self.aggregation_functions_container.remove(x)

    def get_categorical_obs_keys(self, widget=None):
        """Overrides parent function"""
        if self.model is None:
            return []
        return list(self.model.categorical_keys)

    def get_numerical_obs_keys(self, widget=None):
        """Overrides parent function"""
        if self.model is None:
            return []
        return list(self.model.numerical_keys)

    def create_numerical_selection(self, name="Numerical", choices=None):
        if choices is None:
            choices = self.get_numerical_obs_keys()

        return ComboBox(
            name="Numerical", choices=choices, label=name, nullable=True
        )

    def create_category_selection(self, name="Category", choices=None):
        if choices is None:
            choices = self.get_categorical_obs_keys()

        return ComboBox(
            name="Category", choices=choices, label=name, nullable=True
        )

    def create_multi_category_selection(self, name="Category(s)"):
        obs = self.get_categorical_obs_keys()
        Opts = Enum("Obs", obs)
        iterable_obs = list(Opts)
        multi_groupby_selection = create_widget(
            value=[iterable_obs[0]],
            name=name,
            widget_type="ListEdit",
            annotation=list[Opts],
            options={},
        )
        return multi_groupby_selection

    def get_multi_category_selection_choices(self, widget=None):
        return self.parse_enums(self.category_selection.value)

    def get_normalisation_choices(self, widget=None):
        choices = self.parse_enums(self.category_selection.value)
        if (
            len(set(choices)) == 1
        ):  # If one, then normalisation with self with produces nans, remove.
            return []
        return choices

    def local_create_parameter_widgets(self) -> None:
        """Create the specific widgets for each aggregation function."""
        self.clear_local_layout()

        if (
            self.aggregation_functions_selection.value is not None
            and self.sample_key.value is not None
        ):
            aggregation_function = self.aggregation_functions_selection.value

            if aggregation_function == "category_counts":
                self.category_selection = self.create_multi_category_selection(
                    name="Compute for each: "
                )
                self.aggregation_functions_container.extend(
                    [self.category_selection]
                )

            elif aggregation_function == "category_proportions":
                self.category_selection = self.create_multi_category_selection(
                    name="Compute for each: "
                )
                self.category_selection.changed.connect(self.reset_choices)

                self.normalisation_selection = self.create_category_selection(
                    name="Normalise by: ",
                    choices=self.get_normalisation_choices,
                )
                self.aggregation_functions_container.extend(
                    [self.category_selection, self.normalisation_selection]
                )

            elif aggregation_function == "numerical_summarised":
                self.aggregation_function = ComboBox(
                    name="Function: ",
                    choices=[
                        "mean",
                        "sum",
                        "max",
                        "min",
                        "first",
                        "last",
                        "median",
                    ],
                    nullable=False,
                )

                self.numerical_selection = self.create_numerical_selection(
                    name="Variable to aggregate: "
                )

                self.category_selection = self.create_multi_category_selection(
                    name="Compute for each: "
                )
                self.aggregation_functions_container.extend(
                    [
                        self.aggregation_function,
                        self.numerical_selection,
                        self.category_selection,
                    ]
                )

            elif aggregation_function == "numerical_binned":
                self.numerical_selection = self.create_numerical_selection(
                    name="Variable to bin: "
                )

                self.category_selection = self.create_multi_category_selection(
                    name="Compute for each: "
                )
                self.aggregation_functions_container.extend(
                    [self.numerical_selection, self.category_selection]
                )

            else:
                print("Unchcked aggregation function")

            self.sample_adata_selection = ComboBox(
                name="LayersWithContainedAdataSampled",
                choices=self.get_sampling_adata_in_sdata,
                label="Select a sample-level AnnData to put results into",
            )

            self.apply_button = create_widget(
                name="Apply",
                widget_type="PushButton",
                annotation=None,
                options={},
            )
            self.apply_button.changed.connect(self.apply_aggregation_function)
            self.aggregation_functions_container.extend(
                [self.sample_adata_selection, self.apply_button]
            )

    def get_sampling_adata_in_sdata(self, widget=None):
        if self.sdata is not None:
            tables = list(self.sdata.tables.keys())
            # from tables check if .uns "grouping_factor" exists;
            tables_with_grouping = [
                t for t in tables if "grouping_factor" in self.sdata[t].uns
            ]
            sample_key = self.sample_key.value
            if sample_key is not None:
                tables_with_level = [
                    t
                    for t in tables_with_grouping
                    if self.sdata[t].uns["grouping_factor"] == sample_key
                ]
                return tables_with_level
        return []

    def parse_enums(self, enums):
        """i.e. Parse [<Obs.X: 2>] to [X]"""
        return [enum.name for enum in enums]

    def apply_aggregation_function(self):
        """Apply the selected aggregation function."""
        if (
            self.aggregation_functions_selection.value is not None
            and self.model is not None
        ):
            aggregation_function = self.aggregation_functions_selection.value
            if aggregation_function == "category_counts":
                selection = self.parse_enums(self.category_selection.value)
                result = self.model.get_category_counts(
                    categorical_column=selection,
                )
                table_suffix = f"{selection}"

            elif aggregation_function == "category_proportions":
                selection = self.parse_enums(self.category_selection.value)
                normalisation_column = self.normalisation_selection.value
                result = self.model.get_category_proportions(
                    categorical_column=selection,
                    normalisation_column=normalisation_column,
                )
                table_suffix = (
                    f"{selection} normalised by {normalisation_column}"
                )

            elif aggregation_function == "numerical_summarised":
                selection = self.parse_enums(self.category_selection.value)
                numerical_column = self.numerical_selection.value
                aggregation_function = self.aggregation_function.value
                result = self.model.get_numerical_summarised(
                    numerical_column=numerical_column,
                    categorical_column=selection,
                    aggregation_function=aggregation_function,
                )
                table_suffix = (
                    f"{numerical_column} within each " f"{selection}"
                )

            else:
                print("Unchcked aggregation function")

            self.latest_result = result
            self.latest_table = Table(
                value=result,
                # name=
            )
            self.latest_table_label = Label(
                value=f"{aggregation_function} of " + table_suffix
            )
            self.confirm_button = create_widget(
                name="Confirm",
                widget_type="PushButton",
                annotation=None,
                options={},
            )

            table_output = Container(
                widgets=[
                    self.latest_table_label,
                    self.latest_table,
                    self.confirm_button,
                ],
                layout="vertical",
            )
            table_output.native.show()

            self.confirm_button.changed.connect(
                lambda _: _confirm_aggregation(result)
            )
            self.confirm_button.changed.connect(
                lambda _: table_output.native.close()
            )

            def _confirm_aggregation(result):
                sample_factor_name = result.index.name

                in_adata = self.sdata[
                    self.sample_adata_selection.value
                ].copy()  # self.adata.copy()

                # Common order / index
                if in_adata.obs.index.name != sample_factor_name:
                    if sample_factor_name in in_adata.obs.columns:
                        in_adata.obs = in_adata.obs.set_index(
                            sample_factor_name
                        )
                    else:
                        raise ValueError(
                            "Sample factor name not found in AnnData"
                        )

                feature_struct_ordered = result.loc[in_adata.obs.index]
                feature_struct_values = feature_struct_ordered.values
                feature_struct_vars = self.model.get_feature_frame(
                    feature_struct_ordered
                )
                # Check if the i    n_data has existing features
                if (in_adata.X is not None) and (in_adata.var is not None):
                    # # Merge only new features
                    # duplicated_features = feature_struct_vars.index[
                    #     feature_struct_vars.index.isin(in_adata.var.index)
                    # ]

                    # Modify feature matrix
                    merged_X = np.hstack([in_adata.X, feature_struct_values])

                    merged_var = pd.concat(
                        [
                            in_adata.var,
                            feature_struct_vars,
                        ]
                    )
                    merged_var = merged_var.reset_index(drop=True)

                    feature_struct_out = AnnData(
                        obs=in_adata.obs,
                        uns=in_adata.uns,
                        X=merged_X,
                        var=merged_var,
                    )

                else:
                    feature_struct_out = AnnData(
                        obs=in_adata.obs,
                        uns=in_adata.uns,
                        X=feature_struct_values,
                        var=feature_struct_vars,
                    )

                self.sdata[self.sample_adata_selection.value] = (
                    feature_struct_out
                )
                # overwrite_element(
                #     self.sdata, self.sample_adata_selection.value
                # )
                self.events.sdata_changed(sdata=self.sdata)

            # self.aggregation_functions_container.extend(
            #     [
            #         Container(
            #             widgets=[self.latest_table_label, self.latest_table],
            #             layout="vertical"
            #         ),
            #     ]
            # )

        # obs = [NULL_CHOICE] + self.get_obs_keys()

        # Opts = Enum("Obs", obs)
        # iterable_obs = list(Opts)
        # self.groupby_obs_list = create_widget(
        #     value = [
        #         iterable_obs[0]
        #     ],
        #     name="Group By",
        #     widget_type="ListEdit",
        #     annotation=list[Opts],
        #     options={}
        # )
        # self.groupby_obs_list.changed.connect(self.update_groupby_widget)

        # self.measure_obs_list = create_widget(
        #     value = [
        #         iterable_obs[0]
        #     ],
        #     name="Measure By",
        #     widget_type="ListEdit",
        #     annotation=list[Opts],
        #     options={}
        # )
        # self.extend([
        #     self.sample_key,
        #     self.groupby_obs_list,
        #     self.measure_obs_list
        # ])

    def update_local_model(self, key):
        if key is not None:
            self.model = ObsAggregator(self.adata, base_column=key)

    def launch_binning_widget(self):
        pass

    def update_measure_widget(self):
        pass

    def update_groupby_widget(self):
        """If groupby column is numeric, adds a widget for the user to specify
        the number of bins."""

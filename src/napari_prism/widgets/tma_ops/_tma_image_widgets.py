import math
from pathlib import Path

import geopandas as gpd
import napari
import numpy as np
import pandas as pd
import shapely
from magicgui.widgets import ComboBox, Select, Table, create_widget
from napari.qt.threading import thread_worker
from numpy import dtype, ndarray
from qtpy.QtWidgets import QAbstractItemView
from spatialdata.transformations import Scale, Translation, get_transformation
from xarray import DataArray

from napari_prism.models.tma_ops._tma_image import (
    MultiScaleImageOperations,
    SingleScaleImageOperations,
    TMADearrayer,
    TMAMasker,
    TMAMeasurer,
    TMASegmenter,
)
from napari_prism.widgets._widget_utils import (
    create_static_selection_widget,
    get_selected_layer,
)
from napari_prism.widgets.tma_ops._base_widgets import (
    MultiScaleImageNapariWidget,
    SingleScaleImageNapariWidget,
)


class UtilsNapariWidget(MultiScaleImageNapariWidget):
    # TODO:
    # 1) Shapes -> User modifies layer -> Update sdata shapes manually
    # Shift L and Shift E of the layer does not inherit the scaling factor
    # So the added shape is the downscaled one
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: TMAMasker | None = None,  # placeholder?
        *args,
        **kwargs,
    ) -> None:
        super().__init__(viewer, model, *args, **kwargs)
        self.create_parameter_widgets()
        # The model changes when 1) sdata is changed, 2) image_name is changed;
        # These events are trigged by napari_spatialdata.view on layer update
        # When layer is updated -> Get the contained sdata if any;
        self.update_model()
        self.viewer.layers.selection.events.changed.connect(self.update_model)

    def create_parameter_widgets(self):
        self.add_selected_image_channel_button = create_widget(
            value=False,
            name="Add current view as standalone",
            annotation=bool,
            widget_type="PushButton",
            options={"enabled": False},
        )

        self.add_selected_image_channel_button.changed.connect(
            self.add_selected_image_channel
        )

        self.overwrite_save_selected_layer_button = create_widget(
            value=False,
            name="Overwrite save selected layer",
            annotation=bool,
            widget_type="PushButton",
            options={"enabled": False},
        )
        self.overwrite_save_selected_layer_button.changed.connect(
            self.overwrite_save_selected_layer
        )

        self.extend(
            [
                self.add_selected_image_channel_button,
                self.overwrite_save_selected_layer_button,
            ]
        )

    def get_saveable_layers(self, widget=None):
        """Sdata attrs"""
        return [
            x.name
            for x in self.viewer.layers
            if "sdata" in x.metadata and x.metadata["sdata"] is not None
        ]

    def add_selected_image_channel(self):
        channel = self.get_channel()
        image = self.model.get_image()
        image_list = [image.sel(c=channel)[x].image for x in image]
        self.viewer.add_image(
            image_list,
            name=f"{self.model.image_name} View: ({channel})",
            multiscale=True,
            blending="additive",
        )

    def merge_image_labels(
        self,
        labels: list[DataArray | ndarray[dtype[bool]]],
    ) -> DataArray | ndarray[dtype[bool]]:
        """Merge multiple labels into a single DataArray."""

    def update_model(self):
        selected = self.viewer.layers.selection.active

        if selected is not None and "sdata" in selected.metadata:
            if isinstance(selected, napari.layers.shapes.shapes.Shapes):
                # Support only shapes for now
                self.model = SingleScaleImageOperations(
                    selected.metadata["sdata"], selected.metadata["name"]
                )
                self.overwrite_save_selected_layer_button.enabled = True
                self.add_selected_image_channel_button.enabled = False
            elif isinstance(
                selected.data, napari.layers._multiscale_data.MultiScaleData
            ):
                self.model = MultiScaleImageOperations(
                    selected.metadata["sdata"], selected.metadata["name"]
                )
                self.add_selected_image_channel_button.enabled = True
                self.overwrite_save_selected_layer_button.enabled = False

            else:
                self.model = None
                self.add_selected_image_channel_button.enabled = False
                self.overwrite_save_selected_layer_button.enabled = False

            self.reset_choices()

        else:
            self.model = None
            self.reset_choices()
            self.add_selected_image_channel_button.enabled = False
            self.overwrite_save_selected_layer_button.enabled = False

    def overwrite_save_selected_layer(self):
        layer = self.viewer.layers.selection.active
        print("modifiying layer", layer.name)

        # Shapes Directive;
        def _save_annotated_shapes_layer(layer):
            updated_data = layer.data
            annotations_inherited = layer.metadata["_columns_df"][
                "tma_label"
            ].values
            # TODO: Technically below should be handled by the model..?
            geoms_t = [x[:, ::-1] for x in updated_data]
            geoms = [shapely.geometry.Polygon(x) for x in geoms_t]
            gdf_recon = gpd.GeoDataFrame(
                {"geometry": geoms, "tma_label": annotations_inherited}
            )
            # Inherit transformations from layer
            element = layer.metadata["sdata"][layer.metadata["name"]]
            transforms = get_transformation(element, "global")
            print("adding shapes", gdf_recon)

            self.model.add_shapes(
                gdf_recon,
                None,  # Overwrite;
                write_element=True,
                transformations={"global": transforms},
            )

        if (
            isinstance(layer, napari.layers.shapes.shapes.Shapes)
            and "_columns_df" in layer.metadata
            and "sdata" in layer.metadata
            and "name" in layer.metadata
            and isinstance(layer.data, list)
        ):
            _save_annotated_shapes_layer(layer)


class TMAMaskerNapariWidget(MultiScaleImageNapariWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: TMAMasker | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(viewer, model, *args, **kwargs)
        self.create_parameter_widgets()
        # The model changes when 1) sdata is changed, 2) image_name is changed;
        # These events are trigged by napari_spatialdata.view on layer update
        # When layer is updated -> Get the contained sdata if any;
        self.viewer.layers.selection.events.changed.connect(self.update_model)

    #        self.update_model()

    # def set_init_bbox(self):
    #     pass

    def get_bbox_shape_layers(self, widget=None):
        rects = [None]
        if self.model is not None:
            for x in self.viewer.layers:
                if (
                    isinstance(x, napari.layers.shapes.shapes.Shapes)
                    and len(x.shape_type) == 1
                    and x.shape_type[0] == "rectangle"
                ):
                    rects = [None, x.name]
        return rects

    def get_bbox_shape_bounds(
        self, bbox_layer: napari.layers.shapes.shapes.Shapes
    ) -> tuple[int, int, int, int]:
        data_arr = np.array(bbox_layer.data[0])[:, 1:]
        xs = data_arr[:, 1]
        ys = data_arr[:, 0]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        # Cast all values to nearest integer; min as floor, max as ceil
        xmin, ymin = map(math.floor, (xmin, ymin))
        xmax, ymax = map(math.ceil, (xmax, ymax))

        return (xmin, ymin, xmax, ymax)

    def update_model(self):
        # Create widgets on initialisation; can tak from launch method
        # layer = get_selected_layer(
        #     self._viewer, self._select_layer_widget)
        selected = self.viewer.layers.selection.active

        if (
            selected is not None
            and "sdata" in selected.metadata
            and isinstance(
                selected.data, napari.layers._multiscale_data.MultiScaleData
            )
        ):
            self.model = TMAMasker(
                selected.metadata["sdata"], selected.metadata["name"]
            )

            self.reset_choices()
            self._generate_masks_button.enabled = True
        else:
            self.model = None
            self.reset_choices()
            self._generate_masks_button.enabled = False
        # self.set_parent_layer(layer)
        # self.model = TMAMasker(
        #     layer.data,
        #     layer.name,
        #     self.working_channel,
        #     self.scale_index,
        #     layer.metadata["mpp"])

        # TODO: needs a callback to delete the outputs if update_tma_model is called

    def create_parameter_widgets(self):
        super().create_parameter_widgets()

        # self._inherit_correction_settings = create_widget(
        #     value=False,
        #     name="Apply image correction from layer controls",
        #     annotation=bool,
        #     widget_type="CheckBox"
        # )

        # Create widgets on initialisation; can tak from launch method
        self._bbox_shape_layer_selection = ComboBox(
            name="ValidBboxLayers",
            choices=self.get_bbox_shape_layers,
            label="Select shape to mask within",
        )

        self._sigma_slider = create_widget(
            value=10,
            name="Gaussian Blur Sigma/Spread in um",
            annotation=float,
            options={"min": 0, "step": 0.25, "max": 50},
            widget_type="SpinBox",
        )

        self._expansion_slider = create_widget(
            value=10,
            name="Blur expansion in um",
            annotation=int,
            options={"min": 0, "step": 1, "max": 100},
            widget_type="SpinBox",
        )

        self._li_threshold_button = create_widget(
            value=False,
            name="Apply Li Thresholding",
            annotation=bool,
            widget_type="CheckBox",
        )

        self._sobel_button = create_widget(
            value=False,
            name="Apply 2D Sobel filter (edgify)",
            annotation=bool,
            widget_type="CheckBox",
        )

        self._adaptive_histogram_button = create_widget(
            value=False,
            name="Apply local histogram equalisation",
            annotation=bool,
            widget_type="CheckBox",
        )

        self._gamma_correction_button = create_widget(
            value=False,
            name="Apply gamma (0.2) correction",
            annotation=bool,
            widget_type="CheckBox",
        )

        self._estimated_core_diameter_um_entry = create_widget(
            value=700,
            name="Estimated core diameter in um",
            annotation=float,
            widget_type="LineEdit",
        )

        self._thresholding_core_fraction_entry = create_widget(
            value=0.25,
            name="Masks with this fraction of estimated core area to include",
            annotation=float,
            widget_type="LineEdit",
        )

        # TODO: left off
        # self._mask_label_entry = create_widget(
        #     value="mask",
        #     name="Mask label",
        #     annotation=str,
        #     widget_type="LineEdit")

        self._generate_masks_button = create_widget(
            value=False,
            name="Generate masks",
            annotation=bool,
            widget_type="PushButton",
        )

        # Populate container with widgets;
        ls = [
            # self._inherit_correction_settings,
            self._bbox_shape_layer_selection,
            self._sigma_slider,
            self._expansion_slider,
            self._li_threshold_button,
            self._sobel_button,
            self._adaptive_histogram_button,
            self._gamma_correction_button,
            self._estimated_core_diameter_um_entry,
            self._thresholding_core_fraction_entry,
            self._generate_masks_button,
        ]
        # If initialised with an sdata, enable
        if self.model is not None:
            self._generate_masks_button.enabled = True
        else:
            self._generate_masks_button.enabled = False

        self._generate_masks_button.changed.connect(self._generate)
        self.extend(ls)

    # TODO: Thread worker and progress bar
    def _generate(self):  # Similar to nested _generate
        # Check if user has provided and selected a bounding box
        if self._bbox_shape_layer_selection.value is not None:
            bbox_layer = get_selected_layer(
                self.viewer, self._bbox_shape_layer_selection
            )
            bbox_bounds = self.get_bbox_shape_bounds(bbox_layer)
        else:
            bbox_bounds = None

        # Check if user wants to use the viewer settings for the image
        contrast_limits = None
        gamma = None
        # TODO:
        # if self._inherit_correction_settings.value:
        #     selected = self.viewer.layers.selection.active
        #     contrast_limits = selected.contrast_limits
        #     gamma = selected.gamma

        scale = self.get_multiscale_image_scales()[self.scale_index]
        self.model.mask_tma_cores(
            channel=self.get_channel(),
            scale=scale,
            mask_selection=bbox_bounds,
            sigma_um=self._sigma_slider.value,
            expansion_um=self._expansion_slider.value,
            li_threshold=self._li_threshold_button.value,
            edge_filter=self._sobel_button.value,
            adapt_hist=self._adaptive_histogram_button.value,
            gamma_correct=self._gamma_correction_button.value,
            estimated_core_diameter_um=float(
                self._estimated_core_diameter_um_entry.value
            ),
            core_fraction=float(self._thresholding_core_fraction_entry.value),
            contrast_limits=contrast_limits,
            gamma=gamma,
        )

        # refresh the Sdata elements after this is called. NOTE: will be depracated in 0.5.0 napari
        self.refresh_sdata_widget()


class TMADearrayerNapariWidget(SingleScaleImageNapariWidget):
    # TODO: Deleted shape event -> Update layer of tma_labels
    # TODO: Log TMA Grid as an AnnData Tablemodel; obsm attr.
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: TMADearrayer | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(viewer, model, *args, **kwargs)
        self.create_parameter_widgets()
        self.viewer.layers.selection.events.changed.connect(self.update_model)

    def get_line_shape_layers(self, widget=None):
        """Support multiple lines."""
        lines = [None]
        if self.model is not None:
            for x in self.viewer.layers:
                if (
                    isinstance(x, napari.layers.shapes.shapes.Shapes)
                    and x.shape_type[0] == "line"
                ):
                    lines = [None, x.name]
        return lines

    def get_line_mean_lengths(
        self, line_layer: napari.layers.shapes.shapes.Shapes
    ) -> float:
        lines = line_layer.data
        lengths = []
        for line in lines:
            arr = np.array(line)[:, 1:]
            xs = arr[:, 0]
            ys = arr[:, 1]
            length = np.sqrt((xs[1] - xs[0]) ** 2 + (ys[1] - ys[0]) ** 2)
            lengths.append(length)
        return np.mean(lengths)

    def _dearray(self):
        #   layer = get_selected_layer(self.viewer, self._select_layer_widget)
        masks_gdf = self.model.sdata.get(self.model.image_name + "_poly", None)

        self.model.dearray_and_envelope_tma_cores(
            float(self._estimated_core_diameter_um_entry.value),
            expected_rows=int(self._nrows_entry.value),
            expected_cols=int(self._ncols_entry.value),
            expectation_margin=0.2,
            masks_gdf=masks_gdf,
        )  # Sets self.gff

        self.refresh_sdata_widget()
        # TODO: Create coordainte system from bboxes ?

    def _create_tma_grid_widget(self):
        """Create a TMA grid representation of tickboxes; left labels are the A-I of the grid
        bottom labels are 1-10 of the grid.

        If a tick exists, clicking it will remove the core from the representation
        (but its original position kept, so if re-ticked will default to that)

        If a tick doesnt exist, clicking it will add a core in the approximate
        area where it should be (i.e. if E-3 and E-5 exists, if E-4 doesnt exist
        aand then added, then its added between E3 and E4 -> Uses the estimated grid position)

        """
        final_grid_labels = self.model.final_grid_labels
        #        grid_positions = self.model.grid_positions
        rows = len(final_grid_labels)
        cols = len(final_grid_labels[0])
        row_labels = tuple([f"{chr(65 + i)}" for i in range(rows)])
        col_labels = tuple([f"{i+1}" for i in range(cols)])

        grid = Table(final_grid_labels, name="Grid")
        grid.row_headers = row_labels
        grid.column_headers = col_labels

        # Widget resizing
        self.max_width = self.width = 800
        self.max_height = self.height = 1000
        grid.max_width = grid.width = self.max_width
        grid.max_height = grid.height = self.max_height

        # Adjust default spacing
        column_spacing = 20
        for i in range(grid.native.columnCount()):
            grid.native.horizontalHeader().setSectionResizeMode(
                i, grid.native.horizontalHeader().Stretch
            )
            grid.native.setColumnWidth(i, column_spacing)

        # Make read only - noneditable
        grid.native.setEditTriggers(QAbstractItemView.NoEditTriggers)

        self.grid = grid

        def _highlight_selected_tma_cell(r, c):
            print(r, c)

        grid.native.cellClicked.connect(_highlight_selected_tma_cell)

        if "Grid" in self.asdict():
            self.__delattr__("Grid")
        # self.extend([self.grid])

    def _update_tma_grid_widget(self):
        """Update the TMA grid widget - for user-interaction of creating and
        deleting rows. Needs add and remove row in base model class
        TMADearrayer."""

    def update_model(self):
        """TODO: allow for multiple masks -> Mask merge"""
        selected = self.viewer.layers.selection.active
        if (
            selected is not None
            and "sdata" in selected.metadata
            and isinstance(selected, napari.layers.Labels)
        ):
            self.model = TMADearrayer(
                selected.metadata["sdata"], selected.metadata["name"]
            )

            self.reset_choices()
            self._dearray_button.enabled = True
        else:
            self.model = None
            self.reset_choices()
            self._dearray_button.enabled = False

    def create_parameter_widgets(self):
        super().create_parameter_widgets()

        self._estimated_core_diameter_um_entry = create_widget(
            value=700,
            name="Estimated core diameter in um",
            annotation=float,
            widget_type="LineEdit",
        )

        self._nrows_entry = create_widget(
            value=0,
            name="Enforce number of rows",
            annotation=int,
            widget_type="LineEdit",
        )

        self._ncols_entry = create_widget(
            value=0,
            name="Enforce number of columns",
            annotation=int,
            widget_type="LineEdit",
        )

        # Make invisible until mask has been made
        self._dearray_button = create_widget(
            value=False,
            name="Dearray",
            annotation=bool,
            widget_type="PushButton",
            options={"enabled": True},
        )
        self._dearray_button.changed.connect(self._dearray)
        # self._dearray_button.changed.connect(self._create_tma_grid_widget)

        # NOTE: while spatialdata only does cmaps, can use this to manually add labels
        self._output_layer_selection_widget = ComboBox(
            name="OutputLayer",
            choices=self.get_child_shape_layers,
            label="Select output layer",
        )

        self._show_annotations_button = create_widget(
            value=False,
            name="Show annotations",
            annotation=bool,
            widget_type="PushButton",
            options={"enabled": True},
        )
        self._show_annotations_button.changed.connect(self.show_annotations)

        if self.model is not None:
            self._dearray_button.enabled = True
        else:
            self._dearray_button.enabled = False

        self.extend(
            [
                self._estimated_core_diameter_um_entry,
                self._nrows_entry,
                self._ncols_entry,
                self._dearray_button,
                self._output_layer_selection_widget,
                self._show_annotations_button,
            ]
        )

    def get_child_shape_layers(self, widget=None):
        input_layer_name = self._image_layer_selection_widget.value
        if input_layer_name is None:
            return []
        else:
            return [
                x.name
                for x in self.viewer.layers
                if isinstance(x, napari.layers.shapes.shapes.Shapes)
            ]

    def show_annotations(self):
        # NOTE: temp solution for annotation labels
        layer = get_selected_layer(
            self.viewer, self._output_layer_selection_widget
        )

        if (
            "_columns_df" in layer.metadata
            and "tma_label" in layer.metadata["_columns_df"]
        ):
            tma_label_vals = layer.metadata["_columns_df"]["tma_label"]
            layer.features["tma_label"] = tma_label_vals

            layer.text = "tma_label"
            layer.text.visible = True
            layer.text.anchor = "upper_left"
            layer.text.size = 10


class TMASegmenterNapariWidget(MultiScaleImageNapariWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: TMASegmenter | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(viewer, model, *args, **kwargs)
        self.create_parameter_widgets()
        self.viewer.layers.selection.events.changed.connect(self.update_model)

    def update_model(self):
        selected = self.viewer.layers.selection.active

        if (
            selected is not None
            and "sdata" in selected.metadata
            and isinstance(
                selected.data, napari.layers._multiscale_data.MultiScaleData
            )
        ):
            self.model = TMASegmenter(
                selected.metadata["sdata"], selected.metadata["name"]
            )
            self.reset_choices()
            self._run_segmentation_button.enabled = True
        else:
            self.model = None
            self.reset_choices()
            self._run_segmentation_button.enabled = False

    def create_parameter_widgets(self):
        super().create_parameter_widgets()

        self._base_model_selection = create_static_selection_widget(
            "BaseModels",
            TMASegmenter.CP_DEFAULT_MODELS,
            "Cellpose 3.0 Base Models",
        )

        self._denoise_model_selection = create_static_selection_widget(
            "DenoiseModels",
            TMASegmenter.CP_DENOISE_MODELS,
            "(Optional) Cellpose 3.0 Denoiser Models",
        )

        self._custom_model_file_edit = create_widget(
            value=None,
            name="Custom Model File",
            annotation=Path,
            widget_type="FileEdit",
            options={"mode": "r", "enabled": True},
        )
        # NOTE: can rename tiles to 'regions' to be more explicit
        # since tiles can refer to segmentation tiling
        self._tiling_shape_layer_selection = ComboBox(
            name="TilingShapeLayers",
            choices=self.get_tiling_shape_layers,
            label="Select shapes layer for tiling",
            nullable=True,
        )
        self._tiling_shape_layer_selection.changed.connect(self.reset_choices)

        self._tile_name_column_selection = ComboBox(
            name="TileNameColumn",
            choices=self.get_tiling_shape_layer_cols,
            label="Select column for tile names",
        )

        # TODO: incremental like the transforms widget for multichannel + MIP/MaxIP merge then seg on that
        self._segmentation_channel_selection = Select(
            name="SegmentationChannel",
            choices=self.get_channels,
            label="Select channel for segmentation",
        )

        self._channel_merge_method_selection = ComboBox(
            name="ChannelMergeMethods",
            choices=["max", "mean", "sum", "median"],
            label="Method for merging multiple channels",
        )

        self._optional_nuclear_channel_selection = ComboBox(
            name="SegmentationChannel",
            choices=self.get_channels,
            label="Select optional nuclear channel",
            nullable=True,
        )

        # TODO: replace nucleus with just diameter; some models are cyto etc
        self._nuclei_diameter_um_entry = create_widget(
            value=-1,
            name="Nucleus diameter (um)",
            annotation=float,
            widget_type="LineEdit",
            options={
                "tooltip": (
                    "If value is 0 or negative, will default to automated"
                    " diameter estimation."
                )
            },
        )

        self._cellprob_threshold_slider = create_widget(
            value=0.0,
            name="Cell Probability Threshold",
            annotation=float,
            options={"min": -6, "step": 0.01, "max": 6.0},
            widget_type="FloatSlider",
        )

        self._flow_threshold_slider = create_widget(
            value=0.4,
            name="Flow Threshold",
            annotation=float,
            options={"min": 0.0, "step": 0.01, "max": 1.0},
            widget_type="FloatSlider",
        )

        self._debug_mode_button = create_widget(
            value=False,
            name="First and Last Tile only",
            annotation=bool,
            widget_type="CheckBox",
        )

        self._n_random_tiles_entry = create_widget(
            value=0,
            name="Number of random tiles to segment",
            annotation=int,
            widget_type="SpinBox",
            options={"nullable": True},
        )

        # self._preview_segmentation_button = create_widget(
        #     value=False,
        #     name="Preview Segmentation",
        #     annotation=bool,
        #     widget_type="PushButton"
        # )
        # self._preview_segmentation_button.changed.connect(self.preview_segmentation)

        self._run_segmentation_button = create_widget(
            value=False,
            name="Run Segmentation",
            annotation=bool,
            widget_type="PushButton",
        )
        self._run_segmentation_button.changed.connect(self.run_segmentation)
        # self._buttons = QHBoxLayout()
        # self._buttons.addWidget(self._preview_segmentation_button.native)
        # self._buttons.addWidget(self._run_segmentation_button.native)

        self.extend(
            [
                self._base_model_selection,
                self._denoise_model_selection,
                self._custom_model_file_edit,
                self._tiling_shape_layer_selection,
                self._tile_name_column_selection,
                self._segmentation_channel_selection,
                self._channel_merge_method_selection,
                self._optional_nuclear_channel_selection,
                self._nuclei_diameter_um_entry,
                self._cellprob_threshold_slider,
                self._flow_threshold_slider,
                self._debug_mode_button,
                self._run_segmentation_button,
                # self._buttons,
            ]
        )

        # self.native.layout().addLayout(self._buttons)

    def get_tiling_shape_layers(self, widget=None):
        return [
            x.name
            for x in self.viewer.layers
            if isinstance(x, napari.layers.shapes.shapes.Shapes)
            and "sdata" in x.metadata
            and x.name in x.metadata["sdata"].shapes
            and isinstance(
                x.metadata["sdata"].shapes[x.name], gpd.GeoDataFrame
            )
        ]

    def get_tiling_shape_layer_cols(self, widget=None):
        if self._tiling_shape_layer_selection.value is not None:
            layer = get_selected_layer(
                self.viewer, self._tiling_shape_layer_selection.value
            )
            layer_gdf = layer.metadata["sdata"].shapes[layer.name]
            cat_cols = [
                x
                for x in layer_gdf.columns
                if isinstance(
                    layer_gdf[x].dtype, pd.CategoricalDtype | pd.StringDtype
                )
                or pd.api.types.is_string_dtype(layer_gdf[x])
            ]
            return cat_cols
        else:
            return []

    def disable_function_button(self):
        self._run_segmentation_button.enabled = False
        # self._preview_segmentation_button.enabled = False

    def enable_function_button(self):
        self._run_segmentation_button.enabled = True
        # self._preview_segmentation_button.enabled = True

    # def preview_segmentation(self):
    #     self.run_segmentation(preview=True)

    def run_segmentation(self, preview=False):
        worker = self._run_segmentation(preview)
        # TODO: add progress bar; logging message etcd
        worker.start()
        worker.finished.connect(self.post_run_segmentation)
        worker.finished.connect(self.enable_function_button)
        worker.finished.connect(self.refresh_sdata_widget)

        # add_image = partial(
        #     self.viewer.add_image,
        #     name="Segmentation Preview",
        #     rgb=True)

        # worker.returned.connect(add_image)

    def post_run_segmentation(self):
        seg_name = self.model.image_name + "_labels"
        if seg_name in self.viewer.layers:
            labels = get_selected_layer(
                self.viewer, self.model.image_name + "_labels"
            )
            self.viewer.layers.remove(labels)
            sdata_widget = self.get_sdata_widget()
            sdata_widget._onClick(
                text=labels.name
            )  # mimick re-adding the layer

    @thread_worker
    def _run_segmentation(self, preview):
        self.disable_function_button()
        # Validate parameters from widget
        if self._tiling_shape_layer_selection.value is not None:
            # Get the scaling factor of the chosen image scale;
            tiling_layer = get_selected_layer(
                self.viewer, self._tiling_shape_layer_selection
            )

            data_sd = tiling_layer.metadata["sdata"].shapes[tiling_layer.name]
            data_sd = data_sd.copy()
            transforms = get_transformation(data_sd).transformations
            # Scale must be supplied.?
            scale = [x for x in transforms if isinstance(x, Scale)][0].scale
            # Rescale data;
            data_sd["geometry"] = data_sd["geometry"].scale(
                *scale, origin=(0, 0)
            )

            translate = [x for x in transforms if isinstance(x, Translation)]
            if translate != []:
                translate = translate[0].translation
                data_sd["geometry"] = data_sd["geometry"].translate(*translate)

            scale = self.get_multiscale_image_scales()[self.scale_index]
            out = self.model.segment_all(  # noqa: F841
                scale=scale,
                segmentation_channel=self._segmentation_channel_selection.value,
                tiling_shapes=data_sd,
                model_type=self._base_model_selection.value.name,
                nuclei_diam_um=float(self._nuclei_diameter_um_entry.value),
                channel_merge_method=self._channel_merge_method_selection.value,
                optional_nuclear_channel=self._optional_nuclear_channel_selection.value,
                tiling_shapes_annotation_column=self._tile_name_column_selection.value,
                cellprob_threshold=float(
                    self._cellprob_threshold_slider.value
                ),
                flow_threshold=float(self._flow_threshold_slider.value),
                custom_model=self._custom_model_file_edit.value,
                denoise_model=self._denoise_model_selection.value.name,
                debug=self._debug_mode_button.value,
                preview=preview,
            )


class TMAMeasurerNapariWidget(MultiScaleImageNapariWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        model: TMAMeasurer | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(viewer, model, *args, **kwargs)
        self.create_parameter_widgets()
        # The model changes when 1) sdata is changed, 2) image_name is changed;
        # These events are trigged by napari_spatialdata.view on layer update
        # When layer is updated -> Get the contained sdata if any;
        self.viewer.layers.selection.events.changed.connect(self.update_model)

    def create_parameter_widgets(self):
        self._segmentation_layer_selection = ComboBox(
            name="SegmentationLayers",
            choices=self.get_segmentation_layers,
            label="Select segmentation layer",
            nullable=False,
        )
        self._segmentation_layer_selection.changed.connect(self.update_model)

        self._tiling_shape_layer_selection = ComboBox(
            name="TilingShapeLayers",
            choices=self.get_tiling_shape_layers,
            label="Select shapes layer for tiling",
            nullable=True,
        )
        self._tiling_shape_layer_selection.changed.connect(self.reset_choices)

        self._extended_properties_toggle = create_widget(
            value=False,
            name="Measure extended properties",
            annotation=bool,
            widget_type="CheckBox",
            options={
                "tooltip": (
                    f"Additional properties: "
                    f"{TMAMeasurer.EXTENDED_EXPORT_PROPERTIES}"
                )
            },
        )

        self._intensity_mode_selection = ComboBox(
            name="IntensityMode",
            choices=["mean", "median"],
            label="Intensity mode for expression",
            nullable=False,
        )

        self._measure_labels_button = create_widget(
            value=False,
            name="Measure",
            annotation=bool,
            widget_type="PushButton",
            options={
                "tooltip": (
                    f"Measure defaults properties: "
                    f"{TMAMeasurer.EXPORT_PROPERTIES}"
                )
            },
        )
        if self.model is None:
            self._measure_labels_button.enabled = False
        self._measure_labels_button.changed.connect(self.measure_labels)

        self.extend(
            [
                self._segmentation_layer_selection,
                self._tiling_shape_layer_selection,
                self._extended_properties_toggle,
                self._intensity_mode_selection,
                self._measure_labels_button,
            ]
        )

    def enable_function_button(self):
        self._measure_labels_button.enabled = True

    def disable_function_button(self):
        self._measure_labels_button.enabled = False

    def get_segmentation_layers(self, widget=None):
        return [
            x.name
            for x in self.viewer.layers
            if isinstance(x, napari.layers.labels.labels.Labels)
            and "sdata" in x.metadata
            and "adata" in x.metadata
        ]

    def get_tiling_shape_layers(self, widget=None):
        return [
            x.name
            for x in self.viewer.layers
            if isinstance(x, napari.layers.shapes.shapes.Shapes)
            and "sdata" in x.metadata
            and x.name in x.metadata["sdata"].shapes
            and isinstance(
                x.metadata["sdata"].shapes[x.name], gpd.GeoDataFrame
            )
        ]

    def update_model(self):
        # Create widgets on initialisation; can tak from launch method
        # layer = get_selected_layer(
        #     self._viewer, self._select_layer_widget)
        selected = self.viewer.layers.selection.active

        if (
            selected is not None
            and "sdata" in selected.metadata
            and isinstance(
                selected.data, napari.layers._multiscale_data.MultiScaleData
            )
            and self._segmentation_layer_selection.value is not None
        ):
            self.model = TMAMeasurer(
                selected.metadata["sdata"], selected.metadata["name"]
            )

            self.reset_choices()
            self.enable_function_button()
        else:
            self.model = None
            self.reset_choices()
            self.disable_function_button()

    def measure_labels(self):
        worker = self._measure_labels()
        worker.start()
        worker.finished.connect(self.post_measure_labels)

    def post_measure_labels(self):
        # TODO: after this, the labels layer needs to be updated with the new table sdata
        # consider; viewermodel.add_sdata_labels | sdataviewer._get_table_data
        # labels.metadata["sdata"] = self.model.sdata
        # NOTE: temp sol; delete and re-add the layer
        labels = get_selected_layer(
            self.viewer, self._segmentation_layer_selection
        )
        self.viewer.layers.remove(labels)
        sdata_widget = self.get_sdata_widget()
        sdata_widget._onClick(text=labels.name)  # mimick re-adding the layer
        self.enable_function_button()

    @thread_worker
    def _measure_labels(self):
        self.disable_function_button()
        # Validate parameters from widget
        data_sd = None
        if self._tiling_shape_layer_selection.value is not None:
            # Get the scaling factor of the chosen image scale;
            tiling_layer = get_selected_layer(
                self.viewer, self._tiling_shape_layer_selection
            )

            data_sd = tiling_layer.metadata["sdata"].shapes[tiling_layer.name]
            data_sd = data_sd.copy()
            transforms = get_transformation(data_sd).transformations
            scale = [x for x in transforms if isinstance(x, Scale)][0].scale
            # Rescale data;
            data_sd["geometry"] = data_sd["geometry"].scale(
                *scale, origin=(0, 0)
            )

            translate = [x for x in transforms if isinstance(x, Translation)]
            if translate != []:
                translate = translate[0].translation
                data_sd["geometry"] = data_sd["geometry"].translate(*translate)

        labels = get_selected_layer(
            self.viewer, self._segmentation_layer_selection
        )

        self.model.measure_labels(
            labels=labels.data,
            parent_anndata=labels.metadata["adata"],
            exported_table_name=labels.name + "_expression",  # TODO: widget
            tiling_shapes=data_sd,
            extended_properties=self._extended_properties_toggle.value,
            intensity_mode=self._intensity_mode_selection.value,
        )

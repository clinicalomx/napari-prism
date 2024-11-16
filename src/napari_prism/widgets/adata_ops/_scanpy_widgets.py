from collections import defaultdict
from enum import Enum

import napari
import scanpy as sc
from anndata import AnnData
from magicgui.widgets import ComboBox, create_widget
from matplotlib.lines import Line2D
from napari.utils import DirectLabelColormap
from napari.utils.colormaps import label_colormap
from qtpy.QtWidgets import QWidget

from napari_prism import pp, tl
from napari_prism.models.adata_ops._anndata_helpers import ObsHelper
from napari_prism.widgets.adata_ops._base_widgets import AnnDataOperatorWidget
from napari_prism.widgets.adata_ops._plot_widgets import GeneralMPLWidget


class ScanpyClusterCanvas(GeneralMPLWidget):
    """Sub-widget for showing existing scanpy cluster/heatmap-like plots. To be
    used within ScanpyPlotWidget."""

    #: Currently supported scanpy cluster plot functions
    scanpy_clusterplot_funcs = {
        "dotplot": sc.pl.dotplot,
        "matrixplot": sc.pl.matrixplot,
        "stackedviolin": sc.pl.stacked_violin,
        # "clustermap": sc.pl.clustermap,
    }

    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)

    def _plot(
        self,
        adata: AnnData,
        scanpy_plot_func: callable,
        obs_col: str,
        layer: str,
        cmap: str = "Reds",
        vmin: float = None,
        vmax: float = None,
        vcenter: float = None,
        figsize: tuple[int, int] = None,
        with_totals: bool = True,
    ) -> None:
        """Plots a scanpy plot into the widget canvas.

        Args:
            adata: Anndata object
            scanpy_plot_func: Scanpy plot function
            obs_col: Categorical observation column to groupby
            layer: Layer to plot
            cmap: colormap of the cluster/heatmap
            vmin: min value for colormap
            vmax: max value for colormap
            vcenter: center value for colormap
            figsize: figure size
            with_totals: add cell count totals (only for `matrixplot`)
        """
        if with_totals and scanpy_plot_func == sc.pl.matrixplot:
            mp = scanpy_plot_func(
                adata=adata,
                var_names=adata.var_names,
                groupby=obs_col,
                ax=self.axes,
                layer=layer,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                vcenter=vcenter,
                figsize=figsize,
                return_fig=True,
            )
            mp.add_totals().style(edge_color="black").show()

        else:
            scanpy_plot_func(
                adata=adata,
                var_names=adata.var_names,
                groupby=obs_col,
                ax=self.axes,
                layer=layer,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                vcenter=vcenter,
                figsize=figsize,
            )


class ScanpyPlotWidget(AnnDataOperatorWidget):
    """Parent widget for plotting scanpy cluster/heatmap-like plots."""

    def __init__(
        self, viewer: "napari.viewer.Viewer", adata: AnnData, *args, **kwargs
    ) -> None:
        #: Axes for the color map legend
        self.ax_cmap_legend = None
        #: Latest observation selection, used for updating the plot
        self.latest_obs_selection = None
        #: Dictionary mapping categories to colors
        self.cat_to_color = None
        super().__init__(viewer, adata, *args, **kwargs)

        scanpy_plots = [
            "matrixplot",
            "dotplot",
            "stackedviolin",
        ]  # "clustermap"]

        self.scanpy_clusterplot_funcs = {
            "dotplot": sc.pl.dotplot,
            "matrixplot": sc.pl.matrixplot,
            "stackedviolin": sc.pl.stacked_violin,
            # "clustermap": sc.pl.clustermap,
        }
        Opts = Enum("ScanpyPlots", scanpy_plots)
        iterable_opts = list(Opts)
        self.scanpy_plots_list = create_widget(
            value=iterable_opts[0],  # standard
            name="Scanpy Plot Flavor",
            widget_type="ComboBox",
            annotation=Opts,
        )
        self.scanpy_plots_list.scrollable = True
        self.scanpy_plots_list.changed.connect(self.update_plot)

        self.obs_selection = ComboBox(
            name="ObsKeys",
            choices=self.get_categorical_obs_keys,
            label="Select a cat. obs key to groupby",
            value=None,
            nullable=True,
        )
        self.obs_selection.scrollable = True
        self.obs_selection.changed.connect(self.update_plot)
        self.extend([self.scanpy_plots_list, self.obs_selection])

        self.scanpy_canvas = ScanpyClusterCanvas(
            self.viewer, self.native
        )  # FigureCanvas() #FigureCanvasQTAgg()
        # self.toolbar = NavigationToolbar2QT(self.canvas, self.native)
        # self.native.layout().addWidget(self.toolbar)
        self.native.layout().addWidget(self.scanpy_canvas)
        self._expression_selector.changed.connect(
            self.update_plot
        )  # When change expression layer -> update.

        # self.apply_button = create_widget(
        #     name="Color cells by selected key",
        #     widget_type="PushButton",
        #     annotation=bool
        # )
        # self.apply_button.changed.connect(self._run_relabel_cells)
        # self.extend([self.apply_button])

    def _run_relabel_cells(self) -> None:
        """Relabel the colors of cells in the current viewer based on the
        selected categorical observation key. TODO: can depracate, as this
        can be done with View (napari-spatialdata)"""
        worker = self._relabel_cells()
        worker.start()
        # $worker.finished.connect(self.annotate_canvas_with_viewer_cmap)

    # @thread_worker
    def _relabel_cells(self) -> None:
        self.apply_button.native.setEnabled(False)
        if self.obs_selection.value is not None:
            obs = self.adata.obs[self.obs_selection.value]
            cats = obs.unique()
            label = self.adata.obs[self.cell_label_column]
            lbcm = label_colormap(len(cats), background_value=0)
            label_to_cat = dict(zip(label.values, obs.values, strict=False))
            cat_to_color = dict(
                zip(
                    list(cats),
                    [lbcm.colors[x] for x, _ in enumerate(cats)],
                    strict=False,
                )
            )

            lbcm_map = dict(
                zip(
                    label.values,
                    [cat_to_color[label_to_cat[x]] for x in label.values],
                    strict=False,
                )
            )
            lbcm_map = DirectLabelColormap(
                color_dict=defaultdict(int, lbcm_map)
            )
            cell_segmentation_layer = self.get_segmentation_layer()
            cell_segmentation_layer.colormap = lbcm_map
            self.cat_to_color = (
                cat_to_color  # access by plotter -> put legend there
            )

        else:
            cell_segmentation_layer.colormap = (
                cell_segmentation_layer._random_colormap
            )  # Restore defaults
        self.apply_button.native.setEnabled(True)

    def annotate_canvas_with_viewer_cmap(self) -> None:
        """Add a legend to the mpl canvas showing the colors of the cells in
        the viewer."""
        # Add legend to mpl viewer

        if self.cat_to_color is not None:
            self.ax_cmap_legend = True
            self.canvas.axes.legend(
                handles=[
                    Line2D(
                        [0],
                        [0],
                        marker=".",
                        color=c,
                        lw=0,
                        label=b,
                        markerfacecolor=c,
                        markersize=10,
                    )
                    for b, c in self.cat_to_color.items()
                ],
                bbox_to_anchor=(-0.05, 1),
                title="Color in Viewer",
            )

    def update_plot(self) -> None:
        """Update the plot based on the selected observation key and plot
        function."""
        self.latest_obs_selection = self.obs_selection.value

        obs_col = self.obs_selection.value

        if obs_col is not None and obs_col in self.adata.obs.columns:
            if len(obs_col) == 1:
                obs_col = obs_col[0]

            plot_func = self.scanpy_clusterplot_funcs[
                self.scanpy_plots_list.value.name
            ]

            # assume gene-expression like
            cmap = "Reds"  # Scanpy default
            vmin = None
            vmax = None
            vcenter = None
            # figsize = None
            # try figsize heuristic;
            # nrows = self.adata.obs[obs_col].nunique()
            # ncols = self.adata.shape[1]
            # # if nrows == 0:
            # #     nrows = 2
            # BASE_SIZE = 0.25
            # SCALING_FACTOR = 1.5
            # width = BASE_SIZE * ncols
            # height = BASE_SIZE * nrows * SCALING_FACTOR
            # figsize = (width, height)
            # figsize = (ncols // nrows, nrows // 2)
            figsize = None
            # Check bounds
            layer = self._expression_selector.value
            if self.adata.layers[layer].min() < 0:
                cmap = "bwr"
                vmin = self.adata.layers[layer].min()
                vmax = self.adata.layers[layer].max()
                vcenter = 0
                # expression clip at 0

            self.scanpy_canvas.plot(
                adata=self.adata,
                scanpy_plot_func=plot_func,
                obs_col=obs_col,
                layer=layer,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                vcenter=vcenter,
                figsize=figsize,
            )

            if (
                self.ax_cmap_legend
                and self.latest_obs_selection == self.obs_selection.value
            ):  # IF we have a legend, but changed ax, then remake
                self.annotate_canvas_with_viewer_cmap()


class ScanpyFunctionWidget(AnnDataOperatorWidget):
    """Widget for creating widgets to perform scanpy functions on AnnData
    objects."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        super().__init__(viewer, adata)
        self.current_layer = None
        self.gpu = False

    def gpu_toggle(self) -> None:
        self.gpu = not self.gpu
        if self.gpu:
            pp.set_backend("gpu")
            tl.set_backend("gpu")
        else:
            pp.set_backend("cpu")
            tl.set_backend("cpu")

    def update_layer(self, layer) -> None:
        self.current_layer = layer

    def update_model(self, adata: AnnData) -> None:
        self.adata = adata

    def create_parameter_widgets(self) -> None:
        """Has a ComboBox for user to select a scanpy function perform. Calls
        `local_create_parameter_widgets` to create the specific widgets for each
        scanpy function."""
        # NOTE: code golf...
        EMBEDDING_FUNCTIONS = [
            "pca",
            "tsne",
            "neighbors",
            "umap",
            "harmonypy",
        ]

        self.embedding_functions_selection = ComboBox(
            value=None,
            name="Scanpy tl/pp function",
            choices=EMBEDDING_FUNCTIONS,
            nullable=True,
        )
        self.embedding_functions_selection.scrollable = True
        self.embedding_functions_selection.changed.connect(
            self.local_create_parameter_widgets
        )

        self.extend(
            [
                self.embedding_functions_selection,
            ]
        )

    def clear_local_layout(self) -> None:
        """Clear the layout of the scanpy function widgets. Keeps the
        function selection widget."""
        layout = self.native.layout()
        # dont remove the first
        # Remove first item continually until the last
        for _ in range(layout.count() - 1):
            layout.itemAt(1).widget().setParent(None)

    def get_umap_init_pos_choices(self, widget=None) -> list[str]:
        """Get the available choices for the umap init_pos parameter, which can
        use algorithmic initialization or with existing embeddings in .obsm"""
        STATIC = ["spectral", "paga", "random"]
        if self.adata is None:
            return STATIC
        else:
            return STATIC + self.get_obsm_keys()

    def local_create_parameter_widgets(self) -> None:
        """Create the specific widgets for each scanpy function."""
        self.clear_local_layout()

        if self.embedding_functions_selection.value is not None:
            embedding_function = self.embedding_functions_selection.value

            # TODO: easier if using some magicgui decorator
            if embedding_function == "pca":
                self.n_components_entry = create_widget(
                    value=10,
                    options={"min": 1, "step": 1},
                    name="n_comps",
                    annotation=int,
                    widget_type="SpinBox",
                )

                self.extend([self.n_components_entry])

            elif embedding_function == "tsne":
                self.use_rep_selection = ComboBox(
                    value=None,
                    name="use_rep",
                    choices=self.get_obsm_keys,
                    label="Select a representation key",
                    nullable=True,
                )

                self.n_components_entry = create_widget(
                    value=10,
                    options={"min": 1, "step": 1},
                    name="n_pcs",
                    annotation=int,
                    widget_type="SpinBox",
                )

                self.perplexity_entry = create_widget(
                    value=30,
                    options={"min": 1, "step": 1},
                    name="perplexity",
                    annotation=int,
                    widget_type="SpinBox",
                )

                self.early_exaggeration_entry = create_widget(
                    value=12,
                    options={"min": 1, "step": 1},
                    name="early_exaggeration",
                    annotation=int,
                    widget_type="SpinBox",
                )

                self.learning_rate_entry = create_widget(
                    value=1000,
                    options={"min": 1, "step": 1},
                    name="learning_rate",
                    annotation=int,
                    widget_type="SpinBox",
                )

                self.extend(
                    [
                        self.use_rep_selection,
                        self.n_components_entry,
                        self.perplexity_entry,
                        self.early_exaggeration_entry,
                        self.learning_rate_entry,
                    ]
                )

            elif embedding_function == "neighbors":
                self.use_rep_selection = ComboBox(
                    value=None,
                    name="use_rep",
                    choices=self.get_obsm_keys,
                    label="Select a representation key",
                    nullable=True,
                )

                self.n_neighbors_entry = create_widget(
                    value=15,
                    options={"min": 1, "step": 1},
                    name="n_neighbors",
                    annotation=int,
                    widget_type="SpinBox",
                )

                self.n_pcs_entry = create_widget(
                    value=30,
                    options={"min": 1, "step": 1},
                    name="n_pcs",
                    annotation=int,
                    widget_type="SpinBox",
                )

                self.algorithm_entry = ComboBox(
                    value="brute",
                    name="algorithm",
                    choices=["brute", "ivfflat", "ivfpq", "cagra"],
                    label="algorithm",
                )

                self.metric_entry = ComboBox(
                    value="euclidean",
                    name="metric",
                    choices=[
                        "euclidean",
                        "manhattan",
                        "cosine",
                    ],  # truncate to useful ones..
                    label="metric",
                )

                self.extend(
                    [
                        self.use_rep_selection,
                        self.n_neighbors_entry,
                        self.n_pcs_entry,
                        self.algorithm_entry,
                        self.metric_entry,
                    ]
                )

            elif embedding_function == "umap":
                self.min_dist_entry = create_widget(
                    value=0.5,
                    options={"min": 0, "step": 0.1},
                    name="min_dist",
                    annotation=float,
                    widget_type="FloatSpinBox",
                )

                self.spread_entry = create_widget(
                    value=1,
                    options={"min": 0, "step": 0.1},
                    name="spread",
                    annotation=float,
                    widget_type="FloatSpinBox",
                )

                self.alpha_entry = create_widget(
                    value=1,
                    options={"min": 0, "step": 0.1},
                    name="alpha",
                    annotation=float,
                    widget_type="FloatSpinBox",
                )

                self.gamma_entry = create_widget(
                    value=1,
                    options={"min": 0, "step": 0.1},
                    name="gamma",
                    annotation=float,
                    widget_type="FloatSpinBox",
                )

                self.init_pos_entry = ComboBox(
                    value="spectral",
                    name="init_pos",
                    choices=self.get_umap_init_pos_choices,
                    label="init_pos",
                )

                self.extend(
                    [
                        self.min_dist_entry,
                        self.spread_entry,
                        self.alpha_entry,
                        self.gamma_entry,
                        self.init_pos_entry,
                    ]
                )

            elif embedding_function == "harmonypy":
                self.batch_key_selection = ComboBox(
                    name="key",
                    choices=self.get_batch_keys,
                    label="Select a batch key",
                    value=None,
                    nullable=True,
                )
                self.basis_selection = ComboBox(
                    name="basis",
                    choices=self.get_obsm_keys,
                    label="Select a basis key",
                    value=None,
                    nullable=True,
                )
                self.extend([self.batch_key_selection, self.basis_selection])

            else:
                print("Unchecked embedding function")

            self.apply_button = create_widget(
                name="Apply", widget_type="PushButton", annotation=bool
            )
            self.apply_button.changed.connect(self._apply_scanpy_function)
            self.extend([self.apply_button])

    def get_batch_keys(self, widget=None) -> None:
        """Gets obs keys from the AnnData object which may indicate 'batch'
        keys for batch correction. These keys usually have a 1:N relation to
        the cells in the AnnData object (i.e. multiple cells per category).
        """
        if self.adata is None or "index" not in self.adata.obs.columns:
            return []

        else:
            available_batch_keys = list(
                ObsHelper.get_duplicated_keys(self.adata, "index")
            )
            return available_batch_keys

    def collect_parameters(self) -> dict:
        """Collect the parameters from the widgets and return them as a
        dictionary."""
        if self.embedding_functions_selection.value is not None:
            kwargs = {
                widget.name: widget.value
                for widget in self
                if widget.name != "Apply"
            }

            # Append layer kwarg
            if self.current_layer is not None:
                kwargs["layer"] = self.current_layer
                if "use_rep" not in kwargs:
                    kwargs["use_rep"] = self.current_layer

            return kwargs

    def _apply_scanpy_function(self) -> None:
        """Apply the selected scanpy function to the AnnData object."""
        function_map = {
            "pca": tl.pca,
            "tsne": tl.tsne,
            "neighbors": pp.neighbors,
            "umap": tl.umap,
            "harmonypy": tl.harmony,
        }
        scanpy_function = self.embedding_functions_selection.value
        kwargs = self.collect_parameters()
        self.adata = function_map[scanpy_function](
            self.adata, copy=True, **kwargs
        )
        # Refresh widgets subscribing to anndata
        AnnDataOperatorWidget.update_model_all_operators(self.adata)

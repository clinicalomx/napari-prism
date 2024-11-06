import loguru
from typing import TYPE_CHECKING, List, Literal
import magicgui
if TYPE_CHECKING:
    import napari
import decimal
from spatialdata import SpatialData
from magicgui.widgets import create_widget, ComboBox, Select, Table, Container, Label
import squidpy as sq
from anndata import AnnData
import napari
#from napari.layers.utils._link_layers import link_layers
from napari.qt.threading import thread_worker
from napari.utils import DirectLabelColormap
from napari.utils.colormaps import label_colormap
import pandas as pd
from collections import defaultdict
from enum import Enum
from ._widget_utils import get_layer_names, get_layer_index_by_name, \
    gpu_available, make_unique_sdata_element_name # rsc
from ..models.adata_ops.cell_typing._subsetter import AnnDataNodeQT
from ..models.adata_ops.cell_typing._augmentation import add_obs_as_var, subset_adata_by_var
from ..models.adata_ops.cell_typing._preprocessing import AnnDataProcessor, \
    filter_by_obs_count, filter_by_obs_quantile, filter_by_obs_value, \
    filter_by_var_quantile, filter_by_var_value, AnnDataProcessorGPU
from ..models.adata_ops.cell_typing._clustsearch import HybridPhenographSearch, ScanpyClusteringSearch
from ..models.adata_ops.cell_typing._clusteval import ClusteringSearchEvaluator

from ..models.adata_ops._anndata_helpers import ObsHelper
from ._widget_utils import BaseNapariWidget, MultiScaleImageNapariWidget, get_selected_layer, RangeEditFloat, RangeEditInt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QListWidget, QListWidgetItem, QPushButton, QSlider, QLabel, QComboBox,
    QTreeWidget, QTreeWidgetItem, QTableWidget
)
from PyQt5.QtCore import Qt, QTimer# import QtWidgets
from qtpy.QtWidgets import QTabWidget, QAbstractItemView
from kneed import KneeLocator
import scanpy as sc
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, \
    NavigationToolbar2QT, FigureCanvas

# Have it extend napari_matplotlib.
from napari_matplotlib.base import BaseNapariMPLWidget, SingleAxesWidget

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from napari.utils.events import EventedModel, EmitterGroup
from superqt import QLabeledDoubleRangeSlider, QLabeledSlider
from superqt.sliders import MONTEREY_SLIDER_STYLES_FIX
import matplotlib.style as mplstyle

# class MatplotlibHistogramCanvas(QWidget):
#     def __init__(self, data):
#         super().__init__()

#         self.data = data
#         #self.launch_hist()

#     #def launch_hist(sel
#         # Create the matplotlib figure
#         #self.fig, self.ax = plt.subplots()
#         self.fig = Figure(figsize=(4, 7))
#         self.ax = self.fig.add_subplot(111)
#         self.canvas = FigureCanvasQTAgg(self.fig)
#         self.toolbar = NavigationToolbar2QT(self.canvas, self)

#         layout = QVBoxLayout(self)
#         layout.addWidget(self.toolbar)
#         layout.addWidget(self.canvas)

#         # Create sliders
#         self.range_slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
#         self.range_slider.setHandleLabelPosition(
#             QLabeledDoubleRangeSlider.LabelPosition.NoLabel)
    
#         self.range_slider.setStyleSheet(MONTEREY_SLIDER_STYLES_FIX) # macos fix
#         layout.addWidget(self.range_slider)

#         self.nbins_slider = QLabeledSlider(Qt.Horizontal)
#         self.nbins_slider.setRange(0, 500)
#         self.nbins_slider.setValue(0)
#         self.nbins_slider.setStyleSheet(MONTEREY_SLIDER_STYLES_FIX) # macos fix
#         layout.addWidget(QLabel("Number of Bins"))
#         layout.addWidget(self.nbins_slider)

#         self.apply_button = QPushButton("Apply")
#         layout.addWidget(self.apply_button)
#         self.canvas.draw()
#         self.update_hist() # draws

#         # # redl ines
#         # self.lower_line = self.ax.axvline(min_val, color='r')
#         # self.upper_line = self.ax.axvline(max_val, color='r')

#         # Connect sliders to update function
#         self.range_slider.valueChanged.connect(self.update_lines)

#         self.nbins_slider.valueChanged.connect(self.update_hist)

#         # # Initial plot
#         # self.update_lines()
    
#     def update_lines(self):
#          # Get slider values
#         lower_bound, upper_bound = self.range_slider.value()
#         self.lower_vline.set_xdata([lower_bound])
#         self.lower_vline_annot.set_x(lower_bound - 0.01)
#         self.lower_vline_annot.set_text(f"{lower_bound:.2f}")
#         self.upper_vline.set_xdata([upper_bound])      
#         self.upper_vline_annot.set_x(upper_bound + 0.01)  
#         self.upper_vline_annot.set_text(f"{upper_bound:.2f}")
#         # Redraw canvas
#         self.canvas.draw_idle()

#     def update_hist(self):
#         # Retrieve new
#         self.min_val = int(np.floor(min(self.data)))
#         self.max_val = int(np.ceil(max(self.data)))
#         self.range_slider.setRange(self.min_val, self.max_val)
#         self.range_slider.setValue((self.min_val, self.max_val))

#         nbins = self.nbins_slider.value()
#         bins = 'auto' if nbins == 0 else int(nbins)
#         self.ax.clear()
#         self.hist, self.bins, _ = self.ax.hist(
#             self.data, 
#             bins=bins,
#             range=(self.min_val, self.max_val)
#             )
#         lower_bound, upper_bound = self.range_slider.value()
#         self.lower_vline = self.ax.axvline(lower_bound, color='r')
#         self.lower_vline_annot = self.ax.text(
#             lower_bound, 
#             0.99, 
#             f"{lower_bound}",
#             ha="right",
#             va="top",
#             transform=self.ax.get_xaxis_transform(),
#             color="r")
        
#         self.upper_vline = self.ax.axvline(upper_bound, color='r')
#         self.upper_vline_annot = self.ax.text(
#             upper_bound, 
#             0.99, 
#             f"{upper_bound}",
#             ha="left",
#             va="top",
#             transform=self.ax.get_xaxis_transform(),
#             color="r")
#         self.canvas.draw_idle()

#         # TODO: adjust slider steps according to binwidth

class ClusterEvaluatorPlotCanvas(QWidget):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model # Clusteval
        # PARAMS
        # K-R selection -> Each R operates on the same K but not the other way around.

        scores = [
            "Adjusted Rand Index",
            "Normalized Mutual Info",
            "Adjusted Mutual Info",]

        
        Opts = Enum("ClusterScores", scores)
        iterable_opts = list(Opts)
        self.score_list = create_widget(
            value=iterable_opts[0], # standard 
            name="Cluster Scores",
            widget_type="ComboBox",
            annotation=Opts
        )
        self.score_list.scrollable = True
        self.score_list.changed.connect(self.update_plot)

        self.ks_selection = ComboBox(
            name="KParam",
            choices=self.get_ks,
            label="Subset by K parameter",
            nullable=True
        )
        self.ks_selection.scrollable = True
        self.ks_selection.changed.connect(self.update_plot)

        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.score_list.native)
        self.layout.addWidget(self.ks_selection.native)

        self.fig = None
        self.ax = None
        self.canvas = FigureCanvas()#FigureCanvasQTAgg()
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        
        # K, R selection 

        self.update_plot()
    
    def get_ks(self, widget=None):
        if self.model is None:
            return []
        else:
            return self.model.adata.uns["param_grid"]["ks"]

    def update_plot(self):
        if self.fig is not None:
            self.fig.clear()
        if self.ax is not None:
            self.ax.clear()
        self.fig = Figure(figsize=(5, 5))
        self.canvas.figure = self.fig
        self.ax = self.fig.add_subplot(111)

        if self.model is not None:
            score_getters = {
                "Adjusted Rand Index" : self.model.adjusted_rand_index, 
                "Normalized Mutual Info": self.model.normalized_mutual_info, 
                "Adjusted Mutual Info": self.model.adjusted_mutual_info
                }
            k = self.ks_selection.value
            if k is not None:
                k = int(k)
            score_df = score_getters[self.score_list.value.name](k)
            sns.heatmap(score_df, ax=self.ax, cmap="viridis", vmin=0, vmax=1) # Is none, empty plot.
            self.ax.figure.canvas.mpl_connect("pick_event", lambda x: print(x))
        self.canvas.draw_idle()

    def set_new_model(self, model):
        self.model = model
        self.update_plot()

class ClusterEvaluatorTable(QWidget):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model # Clusteval
        self.layout = QVBoxLayout(self)
#        self.table = Table()
        #self.layout.addWidget(self.table.native)

    def set_new_model(self, model):
        self.model = model
        self.update_table()
    
    # def update_table(self):
    #     self.table = Table(self.model.quality_scores)

class AnnDataOperatorWidget(BaseNapariWidget):
    _instances = []
    """ Parent class which operates on an existing AnnData instance. 

        *NEW: AnnData instance, contained within an sdata instance.
    
        Emits events for AnnData changes to trigger callbacks to observing widgets. 

        TODO: instead of adata inputs, works with adata trees; intialised from first adata
    """
    def __init__(
        self, 
        viewer: "napari.viewer.Viewer", 
        adata: AnnData | None = None,
        sdata: SpatialData | None = None,
        *args, 
        **kwargs
    ) -> None:
        super().__init__(viewer, *args, **kwargs)
        self.sdata = sdata
        self.adata = adata
        self.cell_label_column = "index" # This is the default column for cell labels
        self.__class__._instances.append(self)

        if adata is not None:
            self.create_model(adata)

        self._expression_selector = None
        self.create_parameter_widgets()

    def update_model(self, adata):
        self.adata = adata

    def create_model(self, adata):
        self.update_model(adata)
    
    def update_sdata(self, sdata):
        self.sdata = sdata
            
    @classmethod
    def refresh_widgets_all_operators(cls):
        """ If called on its own, does not update model, only widgets. """
#        print(f"Refreshing all operators")
        for instance in cls._instances:
            instance.reset_choices()
    
    @classmethod
    def update_model_all_operators(cls, adata):
        """ Refreshs all operators after model updates. """
#        print(f"Updating all operators with adata: {adata}")
        for instance in cls._instances:
            instance.update_model(adata)
 
        AnnDataOperatorWidget.refresh_widgets_all_operators()

    @classmethod
    def create_model_all_operators(cls, adata):
#        print(f"Creating all operators with adata: {adata}")
        for instance in cls._instances:
            instance.create_model(adata)

        AnnDataOperatorWidget.refresh_widgets_all_operators()

    @classmethod
    def update_sdata_all_operators(cls, sdata):
        for instance in cls._instances:
            instance.update_sdata(sdata)
        
        AnnDataOperatorWidget.refresh_widgets_all_operators()

    def get_obsm_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return list(self.adata.obsm.keys())

    def get_obsp_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return list(self.adata.obsp.keys())
    
    def get_obs_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return list(self.adata.obs.keys())

    def get_categorical_obs_keys(self, widget=None):
        # Can be obs keys too 
        if self.adata is None:
            return []
        else:
            return [
                x for x in self.adata.obs.keys()
                    #if pd.api.types.is_categorical_dtype(self.adata.obs[x])
                    if isinstance(self.adata.obs[x].dtype, pd.CategoricalDtype)
            ]
        
    def get_numerical_obs_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return [
                x for x in self.adata.obs.keys()
                if pd.api.types.is_numeric_dtype(self.adata.obs[x])
            ]
        
    def adata_has_expression(self, adata):
        """ Expression if shape has variables, and obs. """
        n_obs, n_vars = adata.shape
        if n_obs == 0 or n_vars == 0 or adata.X is None:
            return False
        else:
            return True

    def get_expression_layers(self, widget=None):
        if self.adata is None:
            return []
        else:
            if len(self.adata.layers) == 0 \
                and self.adata_has_expression(self.adata):
                self.adata.layers["loaded_X"] = self.adata.X
            return list(self.adata.layers)
    
    def get_expression_and_obsm_keys(self, widget=None):
        # Omit spatial.
        if self.adata is None:
            return []
        else:
            keys = self.get_expression_layers(widget) + self.get_obsm_keys(widget)
            if "spatial" in keys:
                keys.remove("spatial")
            return keys

    def get_markers(self, widget=None):
        # TODO: maybe rename to get_var_keys
        if self.adata is None:
            return []
        else:
            return list(self.adata.var_names)
    
    def reset_choices(self):
        super().reset_choices()
        # set to latest layer
        if self._expression_selector is not None and \
            len(self._expression_selector.choices) > 0:
            self._expression_selector.value = self._expression_selector.choices[-1]

    def create_parameter_widgets(self):
        self._expression_selector = ComboBox(
            name="ExpressionLayers",
            choices=self.get_expression_layers,
            label="Select an expression layer",
        )
        self._expression_selector.scrollable = True
        self.extend([self._expression_selector])

    def get_selected_expression_layer(self):
        expression_layer = self._expression_selector.value
        return self.adata.layers[expression_layer]
    
    def set_selected_expression_layer_as_X(self):
        self.adata.X = self.get_selected_expression_layer()

    def get_segmentation_element(self):
        if self.sdata is None:
            return None
        spatialdata_attrs = self.adata.uns["spatialdata_attrs"]
        seg_element_name = spatialdata_attrs["region"]
        return self.sdata[seg_element_name]
    
    def get_segmentation_layer(self):
        if self.sdata is None:
            return None
        
        spatialdata_attrs = self.adata.uns["spatialdata_attrs"]
        seg_element_name = spatialdata_attrs["region"]
        layer_ix = get_layer_index_by_name(self.viewer, seg_element_name)
        if layer_ix is not None:
            return self.viewer.layers[layer_ix]
        else:
            print(f"Segmentation layer does not exist in viewer")
            return None

    def get_sdata_widget(self):
        #NOTE: private API, temp solution
        # track: https://github.com/scverse/napari-spatialdata/issues/313
        return self.viewer.window._dock_widgets["SpatialData"].widget()
    
    def get_sdata_view_widget(self):
        # NOTE: private API, temp solution
        # track: https://github.com/scverse/napari-spatialdata/issues/313
        return self.viewer.window._dock_widgets["View (napari-spatialdata)"].widget()
    
    def update_layers_with_element(self, element_name):
        """ When overwriting an element on-disk, update layers which have the
            previous in-memory element, with the updated on-disk element. """
        
        # metadata to track;

        # For MSI
        # sdata
        # adata
        # name (element_name)
        # _active_in_cs -> unchanged, since ops dont convert between cs, returns to start
        # _current_cs -> unchanged, as above

        # For Labels
        # actually, if update sdata -> change table annotating layer -> Reads updated table
        # But if table is already selected, need to refresh
        # sdata, as above
        # adata -> can change with annotating layer;
        # indices
        # table_names -> list of annotating tables --> _update_adata calleds
        # 
        pass
    
from napari_matplotlib.util import style_sheet_from_theme

class GeneralMPLWidget(BaseNapariMPLWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)
        self.add_single_axes()
        self.canvas.figure.set_layout_engine("tight")
    
    def plot(self, *args, **kwargs):
        self.clear()
        self.add_single_axes()

        with mplstyle.context(self.napari_theme_style_sheet):
            self._plot(*args, **kwargs)

        self.canvas.draw_idle()

    def clear(self):
        self.figure.clear()
        self.axes.clear()

    def _plot(self):
        print("Abstract method, must be implmented.")

class HistogramPlotCanvas(GeneralMPLWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)
        self.lower_vline = None
        self.lower_vline_annot = None
        self.upper_vline = None
        self.upper_vline_annot = None
        self.data = None
    
    def update_lines(
        self,
        lower_bound: float,
        upper_bound: float,
        lower_bound_label: str,
        upper_bound_label: str
    ) -> None:
        if self.lower_vline is not None:
            self.lower_vline.set_xdata([lower_bound])
            self.lower_vline_annot.set_x(lower_bound - 0.01)
            self.lower_vline_annot.set_text(lower_bound_label)

        if self.upper_vline is not None:
            self.upper_vline.set_xdata([upper_bound])
            self.upper_vline_annot.set_x(upper_bound + 0.01)
            self.upper_vline_annot.set_text(upper_bound_label)

        self.canvas.draw_idle()

    def _plot(
        self,
        data,
        nbins,
        figsize,
        min_val: int | None = None,
        max_val: int | None = None,
        vline_min: float | None = None,
        vline_max: float | None = None,
        vline_min_label: str | None = None,
        vline_max_label: str | None = None,
    ):
        self.canvas.figure.set_figwidth(figsize[0])
        self.canvas.figure.set_figheight(figsize[1])

        # Cache data
        bins = "auto" if nbins == 0 else int(nbins)

        hist, plot_bins, _ = self.axes.hist(
            data,
            bins=bins,
            range=(min_val, max_val)
        )

        if vline_min is not None:
            self.lower_vline = self.axes.axvline(vline_min, color="r")
            self.lower_vline_annot = self.axes.text(
                vline_min,
                0.99,
                vline_min_label,
                ha="right",
                va="top",
                transform=self.axes.get_xaxis_transform(),
                color="r"
            )
        
        if vline_max is not None:
            self.upper_vline = self.axes.axvline(vline_max, color="r")
            self.upper_vline_annot = self.axes.text(
                vline_max,
                0.99,
                vline_max_label,
                ha="left",
                va="top",
                transform=self.axes.get_xaxis_transform(),
                color="r"
            )


class LinePlotCanvas(GeneralMPLWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)
    
    def _plot(
        self,
        *args,
        **kwargs
    ):
        sns.lineplot(
            ax=self.axes
            *args,
            **kwargs
        )

class HeatmapPlotCanvas(GeneralMPLWidget):
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)
    
    def _plot(
        self,
        data,
        cmap,
        vmin,
        vmax,
        vcenter,
        figsize,
        row_cluster: bool = False,
    ):
        sns.clustermap(
            data,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            vcenter=vcenter,
            figsize=figsize,
            ax=self.axes,
            row_cluster=row_cluster
        )

class ScanpyEmbeddingCanvas(GeneralMPLWidget):
    scanpy_embedding_funcs = {
        "umap" : sc.pl.umap, 
        "pca": sc.pl.pca, 
        "tsne": sc.pl.tsne,
    }
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)
    
    def _plot(
        self,
        adata,
        scanpy_embedding_func: callable,
        obs_col: str,
    ):
        scanpy_embedding_func(
            adata,
            color=obs_col,
            annotate_var_explained=True
        )
        
class ScanpyClusterCanvas(GeneralMPLWidget):
    scanpy_clusterplot_funcs = {
        "dotplot" : sc.pl.dotplot, 
        "matrixplot": sc.pl.matrixplot, 
        "stackedviolin": sc.pl.stacked_violin,
        "clustermap": sc.pl.clustermap
    }

    """ Extends napari-matplotlib base class to populate it with existing
        scanpy plotting functions."""
    def __init__(
        self,
        viewer: "napari.viewer.Viewer",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(viewer, parent)

    def _plot(
        self,
        adata,
        scanpy_plot_func: callable,
        obs_col: str,
        layer: str,
        cmap: str = "Reds",
        vmin: float = None,
        vmax: float = None,
        vcenter: float = None,
        figsize: tuple = None,
        with_totals: bool = True
    ):
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
                return_fig=True
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
                figsize=figsize
            )

class ScanpyPlotWidget(AnnDataOperatorWidget):
    """ Base backends are matplotlib.pyplot scatter plots. 
    """
    def __init__(
        self, 
        viewer: "napari.viewer.Viewer", 
        adata: AnnData,
        *args, 
        **kwargs
    ) -> None:
        self.ax_cmap_legend = None
        self.latest_obs_selection = None
        self.cat_to_color = None
        super().__init__(viewer, adata, *args, **kwargs)
        
        # PARAMS
        scanpy_plots = [
            "matrixplot",
            "dotplot",
            "stackedviolin",
            "clustermap"]

        self.scanpy_clusterplot_funcs = {
            "dotplot" : sc.pl.dotplot, 
            "matrixplot": sc.pl.matrixplot, 
            "stackedviolin": sc.pl.stacked_violin,
            "clustermap": sc.pl.clustermap
            }
        Opts = Enum("ScanpyPlots", scanpy_plots)
        iterable_opts = list(Opts)
        self.scanpy_plots_list = create_widget(
            value=iterable_opts[0], # standard 
            name="Scanpy Plot Flavor",
            widget_type="ComboBox",
            annotation=Opts
        )
        self.scanpy_plots_list.scrollable = True
        self.scanpy_plots_list.changed.connect(self.update_plot)
        
        self.obs_selection = ComboBox(
            name="ObsKeys",
            choices=self.get_categorical_obs_keys,
            label="Select a cat. obs key to groupby",
            value=None,
            nullable=True
        )
        self.obs_selection.scrollable = True
        self.obs_selection.changed.connect(self.update_plot)
        self.extend([
            self.scanpy_plots_list,
            self.obs_selection
        ])


        self.scanpy_canvas = ScanpyClusterCanvas(self.viewer, self.native) #FigureCanvas() #FigureCanvasQTAgg()
        #self.toolbar = NavigationToolbar2QT(self.canvas, self.native)
        #self.native.layout().addWidget(self.toolbar)
        self.native.layout().addWidget(self.scanpy_canvas)
        self._expression_selector.changed.connect(self.update_plot) # When change expression layer -> update.

        self.apply_button = create_widget(
            name="Color cells by selected key",
            widget_type="PushButton",
            annotation=bool
        )
        self.apply_button.changed.connect(self._run_relabel_cells)
        self.extend([self.apply_button])

    def _run_relabel_cells(self):
        worker = self._relabel_cells()
        worker.start()
        #$worker.finished.connect(self.annotate_canvas_with_viewer_cmap)

    @thread_worker
    def _relabel_cells(self):
        self.apply_button.native.setEnabled(False)
        if self.obs_selection.value is not None:
            obs = self.adata.obs[self.obs_selection.value]
            cats = obs.unique()
            label = self.adata.obs[self.cell_label_column]
            lbcm = label_colormap(len(cats), background_value=0)
            label_to_cat = dict(zip(label.values, obs.values))
            cat_to_color = dict(zip(list(cats), [lbcm.colors[x] for x,_ in enumerate(cats)]))

            lbcm_map = dict(
                zip(
                    label.values, 
                    [cat_to_color[label_to_cat[x]] for x in label.values]
                    )
                )
            lbcm_map = DirectLabelColormap(color_dict=defaultdict(int, lbcm_map))
            cell_segmentation_layer = self.get_segmentation_layer()
            cell_segmentation_layer.colormap = lbcm_map
            self.cat_to_color = cat_to_color # access by plotter -> put legend there

        else:
            cell_segmentation_layer.colormap = cell_segmentation_layer._random_colormap # Restore defaults
        self.apply_button.native.setEnabled(True)

    def annotate_canvas_with_viewer_cmap(self):
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
                        label=l,
                        markerfacecolor=c,
                        markersize=10,
                        )
                    for l, c in self.cat_to_color.items()
                    ],
                    bbox_to_anchor=(-0.05, 1),
                    title="Color in Viewer"
            )

    def get_categorical_obs_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return [
                x for x in self.adata.obs.keys()
                    if pd.api.types.is_categorical_dtype(self.adata.obs[x])
            ]
        
    def update_plot(self):
        self.latest_obs_selection = self.obs_selection.value

        obs_col = self.obs_selection.value

        if obs_col is not None and obs_col in self.adata.obs.columns:
            if len(obs_col) == 1:
                obs_col = obs_col[0]

            plot_func = self.scanpy_clusterplot_funcs[
                self.scanpy_plots_list.value.name]
            
            # assume gene-expression like
            cmap = "Reds" # Scanpy default
            vmin = None
            vmax = None
            vcenter = None
            #figsize = None
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
                figsize=figsize
            )

            if self.ax_cmap_legend and \
                self.latest_obs_selection == self.obs_selection.value: # IF we have a legend, but changed ax, then remake
                self.annotate_canvas_with_viewer_cmap()

class AnnDataSubsetterWidget(BaseNapariWidget):
    def __init__(
        self, 
        viewer: "napari.viewer.Viewer", 
        adata: AnnData | None = None,
        *args, 
        **kwargs):
        super().__init__(viewer, *args, **kwargs)
        self.adata = adata
        self.events = EmitterGroup(
            source=self, 
            adata_created=None,
            adata_changed=None,
            adata_saved=None)
        
        if adata is not None:
            self.create_model(adata)
        
        self.adata_tree_widget = None
        #self.adata_tree = AnnDataClusterTree(adata) # First initialisation
        self.trees = [] # Track tree instances
        self.create_parameter_widgets()

    def create_model(self, adata, emit=True):
        """ 1) Creation; This creates an entirely new Tree, usually by changing the 
        image parent.
        
        """
        self.adata = adata
        layout = self.native.layout()
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)
        
        if emit:
            self.events.adata_created(value=self.adata)

        self.create_parameter_widgets()

    def update_model(self, adata, emit=True):
        """ Update the Tree; creation / modification"""
        #print(f"Node changed", flush=True)
        #print(adata)
        self.adata = adata
        if emit:
            self.events.adata_changed(value=self.adata)

    def add_anndata_node(self, adata):
        adata_node = AnnDataNodeQT(adata, None, "Root", self.adata_tree_widget)
        for column in range(self.adata_tree_widget.columnCount()):
            self.adata_tree_widget.resizeColumnToContents(column)
        
        self.adata_tree_widget.setCurrentItem(adata_node)

        self.adata_tree_widget.currentItemChanged.connect(
            lambda x: self.update_model(x.adata)
        )
    
    def create_parameter_widgets(self):
        # Export 
        self.save_current_node_button = create_widget(
            name="Export selected AnnData to sdata",
            widget_type="PushButton",
            annotation=bool
        )
        self.save_current_node_button.changed.connect(self.save_current_node)
        self.extend([self.save_current_node_button])
        # QTreeWidget
        self.adata_tree_widget = QTreeWidget()
        self.native.layout().addWidget(self.adata_tree_widget)

        HEADERS = ("AnnData Subset", "Properties")
        self.adata_tree_widget.setColumnCount(len(HEADERS))
        self.adata_tree_widget.setHeaderLabels(HEADERS)
        
        if self.adata is not None:
            self.add_anndata_node(self.adata)

    def add_node_to_current(self, adata_slice, node_label, obs_labels=None):
        # check if node of the same label already exists;
        #TODO: propagate new obs up
        matches = self.adata_tree_widget.findItems(
            node_label, Qt.MatchRecursive, 0)
        
        if matches == []:
            # rTODO: overwrite remove
            AnnDataNodeQT(
                adata_slice,
                obs_labels,
                node_label,
                parent=self.adata_tree_widget.currentItem()
            )
    
    def get_adata_subsets(self):
        pass

    def subset_by_cluster(self, subset, obs_column, cluster_label, var_list=None):
        # Gets the anndata subset from the tree
        # then subsets the anndata to only be of a particular category or
        # cluster_label, from obs_column
        """
        in model:
        def get_cluster_subset(
            self, 
            node_label,
            cluster_subset=None,
            var_list=None):

        """
        pass

    #TODO: this is the most important one; how to save tree-like anndatas..
    def save_current_node(self):
        current_node = self.adata_tree_widget.currentItem()
        current_node.inherit_children_obs()
        adata_out = current_node.adata
        self.events.adata_saved(value=adata_out)

class AugmentationWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        self.events = EmitterGroup(source=self, augment_created=None)
        super().__init__(viewer, adata)
        #self.create_parameter_widgets()

    def reset_choices(self):
        super().reset_choices()
        self.additive_aug.reset_choices()
        self.reductive_aug.reset_choices()

    def create_parameter_widgets(self):
        #super().create_parameter_widgets()
        
        # self.obs_selection = ComboBox(
        #     name="ObsKeys",
        #     choices=self.get_obs_keys,
        #     label="Select a obs key to add as a feature",
        #     value=None,
        #     nullable=True
        # )
        self.augmentation_tabs = QTabWidget()
        self.native.layout().addWidget(self.augmentation_tabs)

        self._expression_selector = ComboBox(
            name="ExpressionLayers",
            choices=self.get_expression_layers,
            label="Select an expression layer",
        )
        self._expression_selector.scrollable = True

        self.obs_selection = Select(
            name="ObsKeys",
            choices=self.get_numerical_obs_keys, # numerical only, string breaks
            label="Select obs keys to add as features",
            value=None,
            nullable=True
        )
        self.obs_selection.scrollable = True
        self.obs_selection.changed.connect(self.reset_choices)

        self.add_obs_as_var_button = create_widget(
            name="Add as feature",
            widget_type="PushButton",
            annotation=bool
        )
        self.add_obs_as_var_button.changed.connect(self._add_obs_as_var)

        self.var_selection = Select(
            name="VarKeys",
            choices=self.get_markers,
            label="Select var keys to subset by",
            value=None,
            nullable=True
        )
        self.var_selection.scrollable = True

        self.subset_var_button = create_widget(
            name="Subset by var",
            widget_type="PushButton",
            annotation=bool
        )
        self.subset_var_button.changed.connect(self._subset_by_var)

        self.additive_aug = Container()
        self.additive_aug.extend([
            self._expression_selector,
            self.obs_selection,
            self.add_obs_as_var_button
        ])

        self.reductive_aug = Container()
        self.reductive_aug.extend([
            self.var_selection,
            self.subset_var_button
        ])

        self.augmentation_tabs.addTab(
            self.additive_aug.native, 
            "Additive Augmentation")
        
        self.augmentation_tabs.addTab(
            self.reductive_aug.native, 
            "Reductive Augmentation")
        
        # self.extend(
        #     [
        #         self.obs_selection,
        #         self.add_obs_as_var_button
        #     ]
        # )

    def get_markers(self, widget=None):
        if self.adata is None:
            return []
        else:
            return list(self.adata.var_names)
    
    def _subset_by_var(self):
        
        var_keys = self.var_selection.value
        if var_keys != []:
            aug_adata = subset_adata_by_var(self.adata, var_keys)
            node_label = f"subset" + "_".join(var_keys)
            self.events.augment_created(value=(aug_adata, node_label))

    def _add_obs_as_var(self):
        
        obs_keys = self.obs_selection.value
        layer_key = self._expression_selector.value
        node_label = "" if layer_key is None else layer_key
        if obs_keys[0] is not None:
            aug_adata = add_obs_as_var(self.adata, obs_keys, layer_key)
            node_label += f"_{'_'.join(obs_keys)}"
            self.events.augment_created(value=(aug_adata, node_label))

    def get_obs_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return [
                x for x in self.adata.obs.keys()
            ]

class QCWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        self.events = EmitterGroup(source=self, augment_created=None)
        super().__init__(viewer, adata)
        self.plot_kwargs = {}
        self.range_slider = None
        self.nbins_slider = None
        self.obs_selection = None
        self.var_selection = None
        self.hist_canvas = None
        self.current_value_directive = None
        self.current_key = "obs"
        self.current_layer = None
    
    def update_layer(self, layer):
        self.current_layer = layer

    def create_parameter_widgets(self):
        self.qc_functions = {
            "filter_by_obs_count": filter_by_obs_count,
            "filter_by_obs_value": filter_by_obs_value,
            "filter_by_obs_quantile": filter_by_obs_quantile,
            "filter_by_var_value": filter_by_var_value,
            "filter_by_var_quantile": filter_by_var_quantile
        }
        Opts = Enum("QCFunctions", list(self.qc_functions.keys()))
        
        self.qc_selection = create_widget(
            value=None,
            name="QC function",
            widget_type="ComboBox",
            annotation=Opts,
            options=dict(
                nullable=True)
        )
        self.qc_selection.scrollable = True
        self.qc_selection.changed.connect(self.local_create_parameter_widgets)

        self.extend([
            self.qc_selection
        ])

    def clear_local_layout(self):
        layout = self.native.layout()
        # dont remove the first
        # Remove first item continually until the last
        for _ in range(layout.count() - 1): 
            layout.itemAt(1).widget().setParent(None)

        if self.hist_canvas is not None:
            self.hist_canvas.clear()

    def create_range_sliders(self):
        self.range_slider = QLabeledDoubleRangeSlider(Qt.Horizontal)
        self.range_slider.setHandleLabelPosition(
            QLabeledDoubleRangeSlider.LabelPosition.NoLabel)

        self.range_slider.setStyleSheet(MONTEREY_SLIDER_STYLES_FIX) # macos fix
        self.range_slider.valueChanged.connect(self.update_lines)
        self.native.layout().addWidget(self.range_slider)

    def create_histogram_plot(self, value_directive="value"):
        # create histogram plot
        self.hist_canvas = HistogramPlotCanvas(
            self.viewer, self.native)
        self.native.layout().addWidget(self.hist_canvas)

    def create_nbins_slider(self):
        self.nbins_slider = QLabeledSlider(Qt.Horizontal)
        self.nbins_slider.setRange(0, 500)
        self.nbins_slider.setValue(0)
        self.nbins_slider.setStyleSheet(MONTEREY_SLIDER_STYLES_FIX) # macos fix
        self.native.layout().addWidget(self.nbins_slider)
        self.nbins_slider.valueChanged.connect(self.on_slider_moved)
        # Buffer the nbins slider so plot isnt updated too frequently
        self.update_timer = QTimer()
        self.update_timer.setInterval(200)  # Delay in milliseconds
        self.update_timer.setSingleShot(True)
        self.update_timer.timeout.connect(self.update_plot)

    def on_slider_moved(self):
        if self.update_timer.isActive():
            self.update_timer.stop()
        self.update_timer.start()
    
    def update_lines(self):
        if self.range_slider is not None:
            min_val, max_val = self.range_slider.value()
            min_val_label = f"{min_val:.2f}" #q
            max_val_label = f"{max_val:.2f}"
            if self.current_value_directive == "quantile":
                if self.current_key == "obs":
                    min_val = np.quantile(
                        self.adata.obs[self.obs_selection.value], min_val)
                    
                    max_val = np.quantile(
                        self.adata.obs[self.obs_selection.value], max_val)
                    
                elif self.current_key == "var":
                    min_val = np.quantile(
                        self.adata[:, self.var_selection.value].layers[self.current_layer], min_val)

                    max_val = np.quantile(
                        self.adata[:, self.var_selection.value].layers[self.current_layer], max_val)
                    

                min_val_label += f" ({min_val:.2f})" #q (value)
                max_val_label += f" ({max_val:.2f})"

            self.hist_canvas.update_lines(min_val, max_val, min_val_label, max_val_label)
    
    def update_plot(
        self, 
    ) -> None:
        if self.adata is not None:
            if self.current_key == "obs":
                if self.obs_selection.value in self.get_categorical_obs_keys():
                    data = self.adata.obs[self.obs_selection.value].value_counts()
                elif self.obs_selection.value in self.get_numerical_obs_keys():
                    data = self.adata.obs[self.obs_selection.value]
                else:
                    raise ValueError("Unchecked obs key")
            
            elif self.current_key == "var":
                adata_sub = self.adata[:, self.var_selection.value]
                if self.current_layer is not None:
                    data = adata_sub.layers[self.current_layer]
                else:
                    if adata_sub.X is not None:
                        print("No expression layer selected. Using .X")
                        data = adata_sub.X
                    else:
                        raise ValueError("Null expression matrices")
            else:
                raise ValueError("Unchecked current_key")
            
            min_val, max_val = int(np.floor(min(data))), int(np.ceil(max(data)))

            if self.current_value_directive == "quantile":
                self.range_slider.setRange(0, 1)
                self.range_slider.setValue((0, 1))
            else:
                self.range_slider.setRange(min_val, max_val)
                self.range_slider.setValue((min_val, max_val))

            nbins = 0 # auto
            if self.nbins_slider is not None:
                nbins = self.nbins_slider.value()
            
            vline_min, vline_max = self.range_slider.value()
        
            vline_min_label = f"{vline_min:.2f}" #q
            vline_max_label = f"{vline_max:.2f}" #q

            if self.current_value_directive == "quantile":
                vline_min = np.quantile(data, vline_min)
                vline_max = np.quantile(data, vline_max)

                vline_min_label += f" ({vline_min:.2f})" #q (value)
                vline_max_label += f" ({vline_max:.2f})" #q (value)

            self.hist_canvas.plot(
                data=data,
                nbins=nbins,
                figsize=(5, 5),
                min_val=min_val,
                max_val=max_val,
                vline_min=vline_min,
                vline_max=vline_max,
                vline_min_label=vline_min_label,
                vline_max_label=vline_max_label
            )
        
    # qc by region -> cell count
    def _apply_qc(self):
        qc_func = self.qc_functions[self.qc_selection.value.name]
        node_label = f"{self.qc_selection.value.name}"
        aug_adata = None

        if self.current_key == "obs":
            obs_key = self.obs_selection.value
            min_val, max_val = self.range_slider.value()
            aug_adata = qc_func(self.adata, obs_key, min_val, max_val)
            if self.obs_selection.value is not None:
                node_label = node_label.replace("obs", self.obs_selection.value)

        else:
            var_key = self.var_selection.value
            min_val, max_val = self.range_slider.value()
            aug_adata = qc_func(self.adata, var_key, min_val, max_val, self.current_layer)
            if self.var_selection.value is not None:
                node_label = node_label.replace("var", self.var_selection.value)

        if aug_adata is not None:
            self.events.augment_created(value=(aug_adata, node_label))

    def local_create_parameter_widgets(self):
        # TODO: left off
        #print("called")
        self.clear_local_layout()

        if self.qc_selection.value is not None:
            # Retrieve qc_selection
            qc_func_selection = self.qc_selection.value.name

            #
            if qc_func_selection == "filter_by_obs_count":
                self.current_value_directive = "value"
                self.current_key = "obs"
                self.obs_selection = ComboBox(
                    name="ObsKeys",
                    choices=self.get_categorical_obs_keys,
                    label="Filter cell populations by obs key",
                )
                self.obs_selection.scrollable = True
                self.obs_selection.changed.connect(self.update_plot)
                self.extend([self.obs_selection])

                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
                
            elif qc_func_selection == "filter_by_obs_value":
                self.current_value_directive = "value"
                self.current_key = "obs"
                self.obs_selection = ComboBox(
                    name="ObsKeys",
                    choices=self.get_numerical_obs_keys,
                    label="Filter cells by obs values",
                )
                self.obs_selection.scrollable = True
                self.obs_selection.changed.connect(self.update_plot)
                self.extend([self.obs_selection])
                
                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
                self.create_nbins_slider()

            elif qc_func_selection == "filter_by_obs_quantile":
                self.current_value_directive = "quantile"
                self.current_key = "obs"
                self.obs_selection = ComboBox(
                    name="ObsKeys",
                    choices=self.get_numerical_obs_keys,
                    label="Filter cells by obs value quantiles",
                )
                self.obs_selection.scrollable = True
                self.obs_selection.changed.connect(self.update_plot)
                self.extend([self.obs_selection])
                
                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
            
            elif qc_func_selection == "filter_by_var_value":
                self.current_value_directive = "value"
                self.current_key = "var"
                self.var_selection = ComboBox(
                    name="VarKeys",
                    choices=self.get_markers,
                    label="Filter cells by var values",
                )
                self.var_selection.scrollable = True
                self.var_selection.changed.connect(self.update_plot)
                self.extend([self.var_selection])
                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
                self.create_nbins_slider()
            
            elif qc_func_selection == "filter_by_var_quantile":
                self.current_value_directive = "quantile"
                self.current_key = "var"
                self.var_selection = ComboBox(
                    name="VarKeys",
                    choices=self.get_markers,
                    label="Filter cells by var value quantiles",
                )
                self.var_selection.scrollable = True
                self.var_selection.changed.connect(self.update_plot)
                self.extend([self.var_selection])
                # create plot elements
                self.create_histogram_plot()
                self.create_range_sliders()
                self.create_nbins_slider()
                
            else:
                print("Unchecked QC function")

            self.apply_button = create_widget(
                    name="Apply QC function",
                    widget_type="PushButton",
                    annotation=bool
                )
            self.apply_button.changed.connect(self._apply_qc)
            self.extend([self.apply_button])

        if self.obs_selection is not None or self.var_selection is not None:
            self.update_plot()

class ScanpyFunctionWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        self.model = None # Let model handle the functions -> handles CPU/GPU switching
        super().__init__(viewer, adata)
        self.current_layer = None
        self.gpu_toggle = None
    
    def update_layer(self, layer):
        self.current_layer = layer

    def update_model(self, adata, model=None):
        self.adata = adata
        if model is not None:
            self.model = model
s
    def create_parameter_widgets(self):
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
            nullable=True
        )
        self.embedding_functions_selection.scrollable = True
        self.embedding_functions_selection.changed.connect(
            self.local_create_parameter_widgets)
        
        if gpu_available():
            self.gpu_toggle = create_widget(
                value=False,
                name="Use GPU",
                annotation=bool
            )
            self.extend([self.gpu_toggle])

        self.extend([
            self.embedding_functions_selection,
        ])

    def clear_local_layout(self):
        layout = self.native.layout()
        # dont remove the first
        # Remove first item continually until the last
        for _ in range(layout.count() - 1): 
            layout.itemAt(1).widget().setParent(None)

    def get_umap_init_pos_choices(self, widget=None):
        STATIC = [
            "spectral",
            "paga",
            "random"
        ]
        if self.adata is None:
            return STATIC
        else:
            return STATIC + self.get_obsm_keys()

    def local_create_parameter_widgets(self):
        self.clear_local_layout()

        if self.embedding_functions_selection.value is not None:
            embedding_function = self.embedding_functions_selection.value
            
            # TODO: easier if using some magicgui decorator 
            if embedding_function == "pca":
                self.n_components_entry = create_widget(
                    value=10,
                    options=dict(
                        min=1, 
                        step=1),
                    name="n_comps",
                    annotation=int,
                    widget_type="SpinBox")
                
                self.extend([
                    self.n_components_entry
                ])
            
            elif embedding_function == "tsne":
                self.n_components_entry = create_widget(
                    value=10,
                    options=dict(min=1, step=1),
                    name="n_pcs",
                    annotation=int,
                    widget_type="SpinBox")
                
                self.perplexity_entry = create_widget(
                    value=30,
                    options=dict(min=1, step=1),
                    name="perplexity",
                    annotation=int,
                    widget_type="SpinBox")

                self.early_exaggeration_entry = create_widget(
                    value=12,
                    options=dict(min=1, step=1),
                    name="early_exaggeration",
                    annotation=int,
                    widget_type="SpinBox")

                self.learning_rate_entry = create_widget(
                    value=1000,
                    options=dict(min=1, step=1),
                    name="learning_rate",
                    annotation=int,
                    widget_type="SpinBox")

                self.extend([
                    self.n_components_entry,
                    self.perplexity_entry,
                    self.early_exaggeration_entry,
                    self.learning_rate_entry
                ])

            elif embedding_function == "neighbors":
                self.n_neighbors_entry = create_widget(
                    value=15,
                    options=dict(min=1, step=1),
                    name="n_neighbors",
                    annotation=int,
                    widget_type="SpinBox")
                
                self.n_pcs_entry = create_widget(
                    value=30,
                    options=dict(min=1, step=1),
                    name="n_pcs",
                    annotation=int,
                    widget_type="SpinBox")
                
                self.algorithm_entry = ComboBox(
                    value="brute",
                    name="algorithm",
                    choices=["brute", "ivfflat", "ivfpq", "cagra"],
                    label="algorithm")

                self.metric_entry = ComboBox(
                    value="euclidean",
                    name="metric",
                    choices=["euclidean", "manhattan", "cosine"], # truncate to useful ones..
                    label="metric")

                self.extend([
                    self.n_neighbors_entry,
                    self.n_pcs_entry,
                    self.algorithm_entry,
                    self.metric_entry
                ])

            elif embedding_function == "umap":
                self.min_dist_entry = create_widget(
                    value=0.5,
                    options=dict(min=0, step=0.1),
                    name="min_dist",
                    annotation=float,
                    widget_type="SpinBox")

                self.spread_entry = create_widget(
                    value=1,
                    options=dict(min=0, step=0.1),
                    name="spread",
                    annotation=float,
                    widget_type="SpinBox")
                
                self.alpha_entry = create_widget(
                    value=1,
                    options=dict(min=0, step=0.1),
                    name="alpha",
                    annotation=float,
                    widget_type="SpinBox")
                
                self.gamma_entry = create_widget(
                    value=1,
                    options=dict(min=0, step=0.1),
                    name="gamma",
                    annotation=float,
                    widget_type="SpinBox")

                self.init_pos_entry = ComboBox(
                    value="spectral",
                    name="init_pos",
                    choices=self.get_umap_init_pos_choices,
                    label="init_pos")

                self.extend([
                    self.min_dist_entry,
                    self.spread_entry,
                    self.alpha_entry,
                    self.gamma_entry,
                    self.init_pos_entry
                ])
    
            elif embedding_function == "harmonypy":
                self.batch_key_selection = ComboBox(
                    name="key",
                    choices=self.get_batch_keys,
                    label="Select a batch key",
                    value=None,
                    nullable=True
                )
                self.basis_selection = ComboBox(
                    name="basis",
                    choices=self.get_obsm_keys,
                    label="Select a basis key",
                    value=None,
                    nullable=True
                )
                self.extend([
                    self.batch_key_selection,
                    self.basis_selection
                ])

            else:
                print("Unchecked embedding function")   

            self.apply_button = create_widget(
                name="Apply",
                widget_type="PushButton",
                annotation=bool
            )
            self.apply_button.changed.connect(self._apply_scanpy_function)
            self.extend([self.apply_button])

    def collect_parameters(self):
        if self.embedding_functions_selection.value is not None:
            kwargs =  {
                widget.name: widget.value for widget in self if widget.name != "Apply"  
            }

            # Append layer kwarg
            if self.current_layer is not None:
                kwargs["layer"] = self.current_layer
                kwargs["use_rep"] = self.current_layer

            return kwargs

    def _apply_scanpy_function(self):
        if self.model is not None:
            scanpy_function = self.embedding_functions_selection.value
            print(scanpy_function)
            kwargs = self.collect_parameters()
            model_func = getattr(self.model, scanpy_function)
            print(model_func)
            model_func(**kwargs)
            # Refresh widgets subscribing to anndata
            AnnDataOperatorWidget.refresh_widgets_all_operators()

# class AnnDataProcessorEmitter(EventedModel, AnnDataProcessor):
#     def __init__(self, adata: AnnData):
#         super().__init__(adata)
# LEFT off; need to sort None type models
class PreprocessingWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        self.events = EmitterGroup(
            source=self, 
            augment_created=None, # Out
            )
        self.model = None
        super().__init__(viewer, adata)
        #self.create_parameter_widgets()

    def create_model(self, adata):
        self.update_model(adata)
        
    def update_model(self, adata):
        self.adata = adata
        self.model = AnnDataProcessor(adata)
        self.embeddings_tab_cls.update_model(self.adata, self.model)
        # if gpu_available():
        #     self.model = AnnDataProcessorGPU(adata)
        # else:
        #     self.model = AnnDataProcessor(adata)

    def reset_choices(self):
        super().reset_choices()
        self.transform_tab.reset_choices()
        self.qc_tab.reset_choices()
    
    def create_parameter_widgets(self):
        super().create_parameter_widgets()
        
        # Expression layer from above; shared
        # self._expression_selector.

        # Processing Tabs
        self.processing_tabs = QTabWidget()
        self.native.layout().addWidget(self.processing_tabs)

        # Transform Tab
        self.transform_tab = Container()
        transforms = [
            "arcsinh",
            "scale",
            "percentile",
            "zscore",
            "log1p"]
        
        # Dispatcher dict for the transform functions
        self.transform_funcs = AnnDataProcessor.transform_funcs
            
        Opts = Enum("Transforms", transforms)
        iterable_opts = list(Opts)
        self.transforms_list = create_widget(
            value=[iterable_opts[0], iterable_opts[1], iterable_opts[-2]], # standard 
            name="Transforms",
            widget_type="ListEdit",
            annotation=List[Opts],
            options=dict(
                tooltip="Arcsinh with cofactor 150, Scale columns and rows to" \
                    "unit variance, 95th percentile normalisation within columns" \
                    "Z-score along rows")
        )
        self.transform_button = create_widget(
            name="Apply",
            widget_type="PushButton",
            annotation=bool
        )
        self.transform_button.changed.connect(self._apply_transforms)
        self.transform_tab.extend(
            [
                self.transforms_list,
                self.transform_button
            ]
        )
        self.processing_tabs.addTab(
            self.transform_tab.native,
            "Transforms"
        )

        # Conditionally create thhis widget based on gpu availability
        # self.gpu_toggle_button = None
        # if gpu_available():
        #     self.gpu_toggle_button = create_widget(
        #         value=False,
        #         name="Use GPU",
        #         annotation=bool
        #     )
        #     self.gpu_toggle_button.changed.connect(self._gpu_toggle)
        #     self.extend([self.gpu_toggle_button])

        # Data QC Tabs
        self.qc_tab = QCWidget(self.viewer, self.adata)
        # ingoing
        self.qc_tab.current_layer = self._expression_selector.value
        self._expression_selector.changed.connect(
            lambda x: self.qc_tab.update_layer(x))
        
        # outgoing
        self.qc_tab.events.augment_created.connect(self.events.augment_created)

        self.processing_tabs.addTab(
            self.qc_tab.native,
            "Quality Control / Filtering"
        )
        
        # self.embeddings_tab = Container()
        # # PCA: -> param: n_components 
        # # TSNE -> also uses n_compoenents, can be used..?
        # # or maybe put this in a tab layout per function..
        # self.pca_n_components_entry = create_widget(
        #     value=10,
        #     options=dict(min=2, step=1),
        #     name="PCA n_components",
        #     annotation=int,
        #     widget_type="SpinBox")
                
        # self.pca_button = create_widget(
        #     name="Run PCA",
        #     widget_type="PushButton",
        #     annotation=bool
        # )
        # self.pca_button.changed.connect(self.run_pca)
        # #TODO: add umap, tsne

        # self.batch_key_selection = ComboBox(
        #     name="BatchKeys",
        #     choices=self.get_batch_keys,
        #     label="Select a batch key",
        # )
        # self.batch_key_selection.scrollable = True
        # self.harmony_button = create_widget(
        #     value=False,
        #     name="(Optional) Harmonypy by provided batch key",
        #     annotation=bool,
        #     widget_type="PushButton")
        # self.harmony_button.changed.connect(self.run_harmony)

        # self.embeddings_tab.extend(
        #     [
        #         self.pca_n_components_entry,
        #         self.pca_button,
        #         self.batch_key_selection,
        #         self.harmony_button
        #     ]
        # )
        # self.processing_tabs.addTab(
        #     self.embeddings_tab.native,
        #     "Embeddings"
        # )

        # test embeddings tab 
        self.embeddings_tab_cls = ScanpyFunctionWidget(self.viewer, self.adata)
        self.embeddings_tab_cls.current_layer = self._expression_selector.value
        self._expression_selector.changed.connect(
            lambda x: self.embeddings_tab_cls.update_layer(x))
        self.processing_tabs.addTab(
            self.embeddings_tab_cls.native,
            "Embeddings"
        )

    def get_batch_keys(self, widget=None):
        if self.model is None or "index" not in self.model.adata.obs.keys():
            return []

        else:
            available_batch_keys = list(ObsHelper.get_duplicated_keys(
                self.model.adata, 
                "index"))
            return available_batch_keys
    
    def _gpu_toggle(self):
        if self.gpu_toggle_button.value == True: 
            self.model.get_CPU_version()
        else:
            self.model = self.model.get_GPU_version()

    @thread_worker
    def _pca(self): # Need to add param for layer 
        self.pca_button.enabled = False
        self.model.pca(
            n_comps=int(self.pca_n_components_entry.value))
        self.pca_button.enabled = True

    def run_pca(self):
        self.set_selected_expression_layer_as_X()
        worker = self._pca()
        worker.start()
        worker.finished.connect(AnnDataOperatorWidget.refresh_widgets_all_operators)

    #@thread_worker # breaks if threaded since harmonypy launches its own workers
    def _harmony(self):
        self.harmony_button.enabled = False
        self.model.harmony(key=self.batch_key_selection.value)
        self.harmony_button.enabled = True

    def run_harmony(self):
        self.set_selected_expression_layer_as_X()
        # worker = self._harmony()
        # worker.start()
        # worker.finished.connect(AnnDataOperatorWidget.refresh_widgets_all_operators)
        self._harmony()
        AnnDataOperatorWidget.refresh_widgets_all_operators()

        # launch the annot widgets?
    def _apply_transforms(self):
        #TODO: left off -> Check transforms_list called.
        self.set_selected_expression_layer_as_X() # adata.layer -> adata.X <<- self.model.adata.X
        transform_label = ""
        for transform in self.transforms_list.value:
            self.model.call_transform(transform.name)
            transform_label += f"{transform.name}_"
        transform_label += self._expression_selector.value # expression layer expression_layer = self._expression_selector.value
        #self.model.adata.layers[transform_label] = self.model.adata.X
        self.adata.layers[transform_label] = self.model.adata.X
        #self.update_model(self.model.adata, changed_model=False)
        #self.events.adata_changed(value=self.model.adata) # or self.adata
        AnnDataOperatorWidget.refresh_widgets_all_operators()


class ClusterSearchWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        super().__init__(viewer, adata)
        #self.create_parameter_widgets()

    def create_model(self, adata):
        self.adata = adata

    def update_model(self, adata):
        self.adata = adata
        self.reset_choices()

    def create_parameter_widgets(self):
        """
        #TODO below gets expressionl layers. With cluster search we can do expression layers and obsms.
            def get_expression_layers(self, widget=None):
                if self.adata.layers == 0:
                    self.adata.layers["loaded_X"] = self.adata.X
                return list(self.adata.layers)
            
            def get_markers(self, widget=None):
                return self.adata.var_names
            
            def create_parameter_widgets(self):
                self._expression_selector = ComboBox(
                    name="ExpressionLayers",
                    choices=self.get_expression_layers,
                    label="Select an AnnData layer",
                    value=self.get_expression_layers()[-1] # Get last in workflow
                )
                self._expression_selector.scrollable = True
                self.extend([self._expression_selector])
        """ 
        # user selects layers
        self.embedding_selector = ComboBox(
            name="EmbeddingLayers",
            choices=self.get_expression_and_obsm_keys,
            label="Select an embedding or expression layer",
        )
        self.embedding_selector.scrollable = True

        CLUSTER_METHODS = ["phenograph", "scanpy"]
        Opts = Enum("ClusterMethods", CLUSTER_METHODS)
        iterable_opts = list(Opts)
        self.cluster_method_list = create_widget(
            value=iterable_opts[0],
            name="Clustering Recipe",
            widget_type="ComboBox",
            annotation=Opts,
        )

        self.knn_range_edit = RangeEditInt(
            start=10,
            stop=30,
            step=5,
            name="K search range for KNN")
        
        self.resolution_range_edit = RangeEditFloat(
            start=0.1,
            stop=1.0,
            step=0.1,
            name="Resolution search range for Leiden Clustering")

        self.min_size_edit = create_widget(
            value=10,
            name="Minimum cluster size",
            annotation=int,
            widget_type="SpinBox",
            options=dict(
                tooltip="If a cluster is found with less than this amount of" \
                    " cells, then that cluster is labelled -1."),
            )

        self.run_param_search_button = create_widget(
            name="Run Parameter Search",
            widget_type="PushButton",
            annotation=bool
        )
        self.run_param_search_button.changed.connect(self.run_param_search_local)

        self.extend(
            [
                self.embedding_selector,
                self.cluster_method_list,
                self.knn_range_edit,
                self.resolution_range_edit,
                self.min_size_edit,
                self.run_param_search_button
            ]
        )
    def get_available_backend(self, cluster_method="phenograph"):
        # If no GPU, enforce all to be CPU
        if cluster_method == "phenograph":
            try:
                import cuml # knn
                import cugraph # dataframe
                return "GPU"
            except ImportError:
                return "CPU"
        elif cluster_method == "scanpy":
            try:
                import rapids_singlecell
                return "GPU"
            except ImportError:
                return "CPU"
        else:
            raise ValueError("Cluster method not recognised.")
    
    def _build_model(self):
        selected_cluster_method = self.cluster_method_list.value.name
        backend = self.get_available_backend(selected_cluster_method)
        if selected_cluster_method == "phenograph":
            self.model = HybridPhenographSearch(
                knn=backend, clusterer=backend) # Refiner left alone due to cpu only
        elif selected_cluster_method == "scanpy":
            self.model = ScanpyClusteringSearch(backend=backend)
        else:
            raise ValueError("Cluster method not recognised.")
    
    @thread_worker
    def _param_search_local(self):
        self.run_param_search_button.enabled = False
        self._build_model()

        # Validate knns
        if self.knn_range_edit.value[0] < 2:
            loguru.logger.warning("KNN minimum less than 2. Setting to 2.")
        
        kes = list(self.knn_range_edit.value)
        kes[1] = kes[1] + kes[2]
        ks = [int(x) for x in np.arange(*kes)]
        #print(ks)

        res = list(self.resolution_range_edit.value)
        res[1] = res[1] + res[2] # Increment stop by step to include stop
        # Round Rs to same decimals as step due to rounding errors in arange
        decimals = decimal.Decimal(str(res[2]))
        est = decimals.as_tuple().exponent * -1 
        rs = [np.round(x, decimals=est) for x in np.arange(*res)]
        #print(rs)

        min_size = int(self.min_size_edit.value)

        # Validate pca has been run
        try:
            self.adata = self.model.parameter_search(
                self.adata,
                embedding_name=self.embedding_selector.value,
                ks=ks,
                rs=rs,
                min_size=min_size)
        
        # pass
        except ValueError as e:
            self.run_param_search_button.enabled = True
            raise ValueError(e) # Just log but set to True
            
        self.run_param_search_button.enabled = True

    def run_param_search_local(self):
        worker = self._param_search_local()
        worker.start()
        worker.finished.connect(AnnDataOperatorWidget.refresh_widgets_all_operators)

    # Experimental; future; model classes for each comp step have Dask Schedulers
    # to submti run jobs on slurm
    # useful for GPU scheduling jobs from an interactive job
    def _param_search_slurm(self):
        pass

    def run_param_search_slurm(self):
        pass

class ClusterAssessmentWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        self.model = None
        self.table = None
        super().__init__(viewer, adata)
        #self.create_parameter_widgets()

    def create_model(self, adata):
        self.update_model(adata)
        
    def update_model(self, adata):
        self.adata = adata

    def update_model_local(self, run_selector_val):
        if self._cluster_run_selector.value is not None:
            self.model = ClusteringSearchEvaluator(
                self.adata, run_selector_val)
        else:
            self.model = None

        self.cc_heatmap.set_new_model(self.model)
        #self.table.set_new_model(self.model)
        #self.add_table()

        self.kr_selection.reset_choices()
        
    def create_parameter_widgets(self):
        self._cluster_run_selector = ComboBox(
            name="ClusterRuns",
            choices=self.get_cluster_runs,
            label="Select a method with a parameter search run",
            nullable=True
        )
        self._cluster_run_selector.scrollable = True
        self._cluster_run_selector.changed.connect(self.update_model_local)
        
        self.extend([self._cluster_run_selector])

        # Plotting Tabs
        self.plot_tabs = QTabWidget()
        self.native.layout().addWidget(self.plot_tabs)

        # All Cluster Runs
        self.cc_heatmap = ClusterEvaluatorPlotCanvas(self.model)
        self._cluster_run_selector.changed.connect(
            self.cc_heatmap.ks_selection.reset_choices) # address in future
        self.plot_tabs.addTab(self.cc_heatmap, "Cluster Score Plots")

        # TODO: pca/umap
        #self.pca_plot = Scan
        # Table
        #self.add_table()

        # K/R selection 
        self.kr_selection = Container(layout="horizontal", labels=True)
        self.k_selection = ComboBox(
            name="KParam",
            choices=self.get_ks,
            label="Select K",
            nullable=True
        )
        self.r_selection = ComboBox(
            name="RParam",
            choices=self.get_rs,
            label="Select R",
            nullable=True
        )
        self.kr_button = create_widget(
            name="Export Cluster Labels to Obs",
            widget_type="PushButton",
            annotation=bool
        )
        self.kr_button.changed.connect(self.add_cluster_to_obs)
        self.kr_selection.extend([
            self.k_selection,
            self.r_selection,
            self.kr_button
        ])
        self.extend([self.kr_selection])

    def add_cluster_to_obs(self):
        if self.k_selection.value is None or self.r_selection.value is None:
            return
        k = int(self.k_selection.value)
        r = float(self.r_selection.value)
        cluster_labels = self.model.get_K_R(k, r).astype("category") # For viewing in obs
        
        self.adata.obs[
            f"{self._cluster_run_selector.value}"
            f"_K{k}"
            f"_R{r}"
            ] = cluster_labels
        
        AnnDataOperatorWidget.refresh_widgets_all_operators()
    
    # def add_table(self):
    #     if self.table is not None and self.plot_tabs.indexOf(self.table) != -1:
    #         self.plot_tabs.removeTab(self.plot_tabs.indexOf(self.table))
    #     self.table = ClusterEvaluatorTable(self.model)
    #     self.plot_tabs.addTab(self.table, "Modularity Scores")

    def get_ks(self, widget=None):
        if self.model is None:
            return []
        else:
            return self.model.adata.uns["param_grid"]["ks"]
    
    def get_rs(self, widget=None):
        if self.model is None:
            return []
        else:
            return self.model.adata.uns["param_grid"]["rs"]

    def get_cluster_runs(self, widget=None):
        searchers = ClusteringSearchEvaluator.IMPLEMENTED_SEARCHERS
        
        available_runs = []
        if self.adata is None:
            return available_runs
        else:

            for searcher in searchers:
                if searcher+"_labels" in self.adata.obsm:
                    available_runs.append(searcher)
            
            return available_runs


import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLineEdit, QHeaderView, QTableWidgetItem, QTableView, QInputDialog
from PyQt5.QtCore import Qt
from magicgui.widgets import Table
import pandas as pd

class EditableTable(Table):
    """ Hack by making headers the first row, then hiding headers. """
    def __init__(self, value, *args, **kwargs):
        value = self.drop_key_to_val(value)
        super().__init__(value=value, *args, **kwargs)
        self.init_ui()

    @staticmethod
    def drop_key_to_val(value):
        new_d = {}
        for i, item in enumerate(value.items()):
            k, v = item
            new_list = [k] + v # [key, labels, ...]
            new_d[i] = new_list

        return new_d

    @staticmethod
    def reverse_drop_key_to_val(value):
        original = [x[0] for x in value] # first col
        new = [x[1] for x in value] # rest
        reverse_d = {}
        for v in [original, new]:
            reverse_d[v[0]] = v[1:]

        return reverse_d

    def init_ui(self):
        # Make second one editable
        #self.table.horizontalHeader().sectionDoubleClicked.connect(self.changeHorizontalHeader)

        # Hide the vertical header
        self.native.verticalHeader().setVisible(False)

        # Hide the horizontal header
        self.native.horizontalHeader().setVisible(False)

        # Make the first row header like; align centre
        for col in range(self.native.columnCount()):
            frow_item = self.native.item(0, col)
            frow_item.setTextAlignment(Qt.AlignHCenter)

        # Make the first column uneditable
        for row in range(self.native.rowCount()):
            item = self.native.item(row, 0)
            if item is None:
                item = QTableWidgetItem()
                self.native.setItem(row, 0, item)
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)

class ClusterAnnotatorWidget(AnnDataOperatorWidget):
    """ Special case, this has to communicate with the tree widget. """
    DEFAULT_ANNOTATION_NAME = "Annotation"
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        self.annotation_table = None
        super().__init__(viewer, adata)
        # self.create_parameter_widgets()

    def create_model(self, adata):
        self.update_model(adata)
        
    def update_model(self, adata):
        self.adata = adata

    def create_parameter_widgets(self):
        # Retrieve the inspection widget canvas; NOTE, revise this design choice...
        # self.obs_widget = self.processor_parent.inspector.obs_widget
        self.obs_widget = ScanpyPlotWidget(self.viewer, self.adata)
        # self.obs_selection = ComboBox(
        #     name="ObsKeys",
        #     choices=self.get_categorical_obs_keys,
        #     label="Select a cat. obs key to annotate",
        #     value=None,
        #     nullable=True
        # )
        # self.obs_selection.scrollable = True
        # self.obs_selection.changed.connect(self.get_init_table)
        self.obs_selection = self.obs_widget.obs_selection
        self.obs_selection.changed.connect(self.get_init_table)
        self.extend([self.obs_widget])

    def get_categorical_obs_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return [
                x for x in self.adata.obs.keys()
                    #if pd.api.types.is_categorical_dtype(self.adata.obs[x])
                    if isinstance(self.adata.obs[x].dtype, pd.CategoricalDtype)
            ]
    
    def get_init_table(self, widget=None):
        if self.annotation_table is not None:
            self.remove(self.annotation_table)
        
        if self.obs_selection.value is not None:
            label_name = self.obs_selection.value
            labels = sorted(list(self.adata.obs[label_name].unique()))
            tbl = {
                label_name: labels,
                self.DEFAULT_ANNOTATION_NAME: [None] * len(labels) # Make header editable
                }
            
            self.annotation_table = EditableTable(
                tbl,
                name="Annotation Table"
            )
            self.annotation_table.changed.connect(self.update_obs_mapping) # Or connect to callback button
            # TODO: on header edit -> Update obs column
            # TODO: on annotation cell edit -> Update obs mapping
            
            self.extend([self.annotation_table])

    def update_obs_mapping(self):
        value = self.annotation_table.value["data"]
        d = EditableTable.reverse_drop_key_to_val(value)
        original_obs, original_labels = list(d.items())[0]
        new_obs, new_labels = list(d.items())[1]
        if self.DEFAULT_ANNOTATION_NAME != new_obs: # Relabeled it
            if self.DEFAULT_ANNOTATION_NAME in self.adata.obs:
                del self.adata.obs[self.DEFAULT_ANNOTATION_NAME]

        self.adata.obs[new_obs] = self.adata.obs[original_obs].map(
            dict(zip(original_labels, new_labels)))
        self.adata.obs[new_obs] = self.adata.obs[new_obs].astype("category")
        AnnDataOperatorWidget.refresh_widgets_all_operators()

    def get_obs_categories(self, widget=None):
        if self.obs_selection.value is not None:
            if self.obs_selection.value[0] is not None:
                obs = self.adata.obs[self.obs_selection.value[0]]
                return obs.unique()
        return []


class SubclusteringWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        self.events = EmitterGroup(source=self, subcluster_created=None)
        super().__init__(viewer, adata)
        #self.create_parameter_widgets()

    def create_model(self, adata):
        self.update_model(adata)
        
    def update_model(self, adata):
        self.adata = adata

    def create_parameter_widgets(self):
        self.obs_selection = ComboBox(
            name="ObsKeys",
            choices=self.get_categorical_obs_keys,
            label="Select a cat. obs key to annotate",
            value=None,
            nullable=True
        )
        self.obs_selection.scrollable = True
        self.obs_selection.changed.connect(self.reset_choices)

        self.obs_label_selection = ComboBox(
            name="ObsCategories",
            choices=self.get_obs_categories,
            label="Select a category to subcluster",
            value=None,
            nullable=True
        )
        self.obs_label_selection.scrollable = True

        self.var_selection = Select(
            name="VarKeys",
            choices=self.get_markers,
            label="Select a marker to visualise",
            value=None,
            nullable=True
        )
        self.var_selection.scrollable = True

        #TODO; this button doesnt work?
        self.subcluster_button = create_widget(
            name="Subcluster",
            widget_type="PushButton",
            annotation=bool
        )
        self.subcluster_button.changed.connect(self.subcluster)

        self.extend([
            self.obs_selection, 
            self.obs_label_selection, 
            self.var_selection,
            self.subcluster_button])

    def subcluster(self):
        #print("create subcluster pushed", flush=True)
        adata_subset = self.adata.copy()
        label = None
        var_suffix = ""
        # Query by obs
        if self.obs_selection.value is not None:
            obs = self.obs_selection.value
            label = self.obs_label_selection.value
            adata_subset = adata_subset[adata_subset.obs[obs] == label].copy()
        
        # Query by var
        if self.var_selection.value is not None:
            adata_subset = adata_subset[:, self.var_selection.value].copy()
            var_suffix = "_truncated_markers"

        # Add to tree
        node_label = label if label is not None else obs+var_suffix
        
        self.events.subcluster_created(value=(adata_subset, node_label))
        # self.subsetter.add_node_to_current(
        #     adata_subset,
        #     node_label=node_label)
        

    def get_markers(self, widget=None):
        if self.adata is None:
            return []
        else:
            return list(self.adata.var_names)
    
    def get_categorical_obs_keys(self, widget=None):
        if self.adata is None:
            return []
        else:
            return [
                x for x in self.adata.obs.keys()
                    if pd.api.types.is_categorical_dtype(self.adata.obs[x])
            ]
    
    def get_obs_categories(self, widget=None):
        if self.obs_selection.value is not None:
            obs = self.adata.obs[self.obs_selection.value]
            return obs.unique()
        return []

class CellTypingTab(QTabWidget):
    """ UI tabs. """
    def __init__(self, viewer: "napari.viewer.Viewer", adata, subsetter):
        super().__init__()
        self.viewer = viewer
        self.subsetter = subsetter
        
        # Cell Typing
        # self.qc = QualityControlWidget(self.viewer, adata)
        # self.qc.max_height = 700
        # self.tabs.addTab(self.qc.native, "Quality Control")

        # self.inspector = InspectionWidget(self.viewer, adata)
        # self.inspector.max_height = 700
        # self.addTab(self.inspector.native, "Inspect AnnData")
        
        self.augmentation = AugmentationWidget(self.viewer, adata)
        self.augmentation.max_height = 400
        self.augmentation.events.augment_created.connect(
            lambda x: self.subsetter.add_node_to_current(
                x.value[0], 
                node_label=x.value[1])
            )
        self.addTab(self.augmentation.native, "Augmentation")

        self.preprocessor = PreprocessingWidget(self.viewer, adata)
        self.preprocessor.max_height = 900
        self.preprocessor.events.augment_created.connect(
            lambda x: self.subsetter.add_node_to_current(
                x.value[0], 
                node_label=x.value[1])
            )
        self.addTab(self.preprocessor.native, "Preprocessing")
        
        self.clustering_searcher = ClusterSearchWidget(self.viewer, adata)
        self.clustering_searcher.max_height = 400
        self.addTab(self.clustering_searcher.native, "Clustering Search")

        self.cluster_assessment = ClusterAssessmentWidget(self.viewer, adata)
        self.cluster_assessment.max_height = 700
        self.addTab(self.cluster_assessment.native, "Assess Cluster Runs")

        self.cluster_annotator = ClusterAnnotatorWidget(
            self.viewer, adata)
        self.cluster_annotator.max_height = 900
        self.addTab(self.cluster_annotator.native, "Annotate Clusters")

        # Needs root access
        self.subclusterer = SubclusteringWidget(
            self.viewer, adata)
        self.subclusterer.max_height = 700
        self.subclusterer.events.subcluster_created.connect(
            lambda x: self.subsetter.add_node_to_current(
                x.value[0], 
                node_label=x.value[1])
            )
        self.addTab(self.subclusterer.native, "Subclusterer")

class GraphBuilderWidget(AnnDataOperatorWidget):
    DEFAULT_STATIC_KWARGS = {
        "elements_to_coordinate_systems": None, # Widgets work with Adatas, Sdata inputs not supported for the moment
        "table_key": None, # Widgets work with Adatas, Sdata inputs not supported for the moment
        "coord_type": "generic", # Imaging package, not visium/etc
    }
    """ Wrapper for sq.gr.spatial_neighbors functions + extra utils.
        TODO: improve, alot of boilerplate
    """
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        super().__init__(viewer, adata)
    
    def create_parameter_widgets(self):
        # Parameters for spatial_neighbors; can probably do magicgui deco ? 
        # no -> Need to dynamically get parameters with getter methods

        # sq.spatial_neighbors parameters
        self.spatial_key = ComboBox(
            name="SpatialKeys",
            choices=self.get_obsm_keys,
            label="spatial_key")
        self.spatial_key.scrollable = True

        self.library_key = ComboBox(
            name="LibraryKeys",
            choices=self.get_categorical_obs_keys,
            label="library_key")
        self.library_key.scrollable = True

        self.n_neighs = create_widget(
            value=10,
            name="n_neighs",
            annotation=int,
            widget_type="SpinBox",
            options=dict(
                min=1,
                max=200,
                step=1)
        )

        self.radius = create_widget(
            value=-1,
            name="radius",
            annotation=float,
            widget_type="SpinBox",
            options=dict(
                min=-1,
                step=1.0,
                nullable=True)
        )

        self.delaunay = create_widget(
            value=False,
            name="delaunay",
            widget_type="CheckBox",
            annotation=bool
        )

        self.percentile = create_widget(
            value=99,
            name="percentile",
            annotation=float,
            widget_type="SpinBox",
            options=dict(
                min=0.0,
                max=100.0,
                step=1.0,
                nullable=True)
        )

        self.transform = ComboBox(
            value=None,
            name="transform",
            choices=["spectral", "cosine"],
            label="transform",
            nullable=True
        )

        self.set_diag = create_widget(
            value=False,
            name="set_diag",
            widget_type="CheckBox",
            annotation=bool
        )
        
        self.key_added = create_widget(
            value="spatial",
            name="key_added",
            annotation=str,
            widget_type="LineEdit"
        )

        self.build_graph_button = create_widget(
            name="Build Graph",
            widget_type="PushButton",
            annotation=bool
        )
        self.build_graph_button.changed.connect(self.build_graph)

        self.extend(
            [
                self.spatial_key,
                self.library_key,
                self.n_neighs,
                self.radius,
                self.delaunay,
                self.percentile,
                self.transform,
                self.set_diag,
                self.key_added,
                self.build_graph_button
            ]
        )

    @thread_worker
    def _build_graph(self):
        self.build_graph_button.enabled = False
        kwargs = self.DEFAULT_STATIC_KWARGS
        kwargs["spatial_key"] = self.spatial_key.value
        kwargs["library_key"] = self.library_key.value
        kwargs["n_neighs"] = self.n_neighs.value
        kwargs["radius"] = self.radius.value if self.radius.value > -1 else None
        kwargs["delaunay"] = self.delaunay.value
        kwargs["percentile"] = self.percentile.value
        kwargs["transform"] = self.transform.value
        kwargs["set_diag"] = self.set_diag.value
        kwargs["key_added"] = self.key_added.value

        # Inplace operations.
        sq.gr.spatial_neighbors(self.adata, copy=False, **kwargs)
        self.build_graph_button.enabled = True

    def build_graph(self):
        worker = self._build_graph()
        worker.start()
        worker.finished.connect(AnnDataOperatorWidget.refresh_widgets_all_operators)

from tmaprocessor.models.adata_ops.spatial_analysis._cell_level import \
    cellular_neighborhoods, cellular_neighborhoods_sq
class NolanComputeWidget(AnnDataOperatorWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        super().__init__(viewer, adata)

    def create_parameter_widgets(self):
        self.connectivity_key = ComboBox(
            name="SpatialKeys",
            choices=self.get_obsp_keys,
            label="spatial_key")
        self.connectivity_key.scrollable = True

        self.library_key = ComboBox(
            name="LibraryKeys",
            choices=self.get_categorical_obs_keys,
            label="library_key")
        self.library_key.scrollable = True

        self.phenotype_key = ComboBox(
            name="PhenotypeKeys",
            choices=self.get_categorical_obs_keys,
            label="phenotype"
        )

        self.k_kmeans_selection = RangeEditInt(
            start=5,
            stop=15,
            step=1,
            name="Number of CNs to search")
        
        self.mini_batch_kmeans_toggle = create_widget(
            value=False,
            name="mini batch kmeans",
            widget_type="CheckBox",
            annotation=bool
        )

        # self.parallelise_toggle = create_widget(
        #     value=False,
        #     name="parallelise",
        #     widget_type="CheckBox",
        #     annotation=bool
        # )

        self.compute_cns_button = create_widget(
            name="Compute CNs",
            widget_type="PushButton",
            annotation=bool
        )
        self.compute_cns_button.changed.connect(self.compute_nolan_cns)

        #TODO left off;
        #self.enrichment_heatmap = pass

        self.extend(
            [
                self.connectivity_key,
                #self.library_key,
                self.phenotype_key,
                self.k_kmeans_selection,
                self.mini_batch_kmeans_toggle,
                self.compute_cns_button
            ]
        )

    @thread_worker
    def _compute_nolan_cns(self):
        self.compute_cns_button.enabled = False 
        kes = list(self.k_kmeans_selection.value)
        kes[1] = kes[1] + kes[2]
        ks = [int(x) for x in np.arange(*kes)]

        cellular_neighborhoods_sq(
            self.adata,
            phenotype=self.phenotype_key.value,
            connectivity_key=self.connectivity_key.value,
            #library_key=self.library_key.value,
            k_kmeans=ks,
            mini_batch_kmeans=self.mini_batch_kmeans_toggle.value,
        )
        
        self.compute_cns_button.enabled = True

    def compute_nolan_cns(self):
        worker = self._compute_nolan_cns()
        worker.start()
        worker.finished.connect(
            AnnDataOperatorWidget.refresh_widgets_all_operators)

    def get_nolan_enrichment(self):
        pass

class NolanPlotWidget(QTabWidget):
    CN_INERTIAS_KEY = "cn_inertias"
    CN_INERTIAS_ATTR = "uns"
    CN_ENRICHMENT_KEY = "cn_enrichment_matrices"
    CN_ENRICHMENT_ATTR = "uns"
    CN_LABELS_KEY = "cn_labels"
    CN_LABELS_ATTR = "obsm"

    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        super().__init__()
        self.viewer = viewer
        self.adata = adata
        # PLOTS
        nolan_plots = [
            "kneedle",
            "enrichment",
            ""
        ]
    # self.kneedle_plot
        self.knee_point = None
        self.kneedle_plot = LinePlotCanvas(self.viewer, self)
        self.addTab(self.kneedle_plot, "Kneedle Plot")

        self.enrichment_matrix_plot = Container()
        self.enrichment_matrix_canvas = HeatmapPlotCanvas(
            self.viewer, self)
        self.choose_K = ComboBox(
            name="Ks",
            choices=self.get_k_kmeans,
            label="Choose K Kmeans run",
            value=None,
            nullable=True
        )
        self.choose_K.changed.connect(self.update_enrichment_plot)
        self.enrichment_matrix_plot.extend([self.choose_K])
        self.enrichment_matrix_plot.native.layout().addWidget(
            self.enrichment_matrix_canvas
        )
        self.addTab(self.enrichment_matrix_plot.native, "Enrichment Matrix")

    def reset_choices(self):
        super().reset_choices()
        self.update_kneedle_plot()

    def update_kneedle_plot(self):
        uns_d = getattr(self.adata, self.CN_INERTIAS_ATTR)

        inertia_results = uns_d.get(self.CN_INERTIAS_KEY, None)

        if inertia_results is not None:
            ks = inertia_results.index
            inertias = inertia_results.values
            kneedle = KneeLocator(
                x=list(ks),
                y=list(inertias.flatten()),
                S=1.0,
                curve="convex",
                direction="decreasing"
            )
            self.knee_point = kneedle.knee
            self.kneedle_plot.plot(
                inertia_results.reset_index(),
                x="k_kmeans",
                y="Inertia"
            )
            self.kneedle_plot.axes.axvline(
                kneedle.knee,
                linestyle="--",
                label="knee/elbow",
            )
            self.kneedle_plot.axes.legend()
        else:
            self.kneedle_plot.clear()

    def get_k_kmeans(self, widget=None):
        if self.adata is None:
            return []
        else:
            uns_d = getattr(self.adata, self.CN_INERTIAS_ATTR)

            inertia_results = uns_d.get(self.CN_INERTIAS_KEY, None)

            if inertia_results is None:
                return []
            else:
                return list(inertia_results.index)

    def update_enrichment_plot(self):
        uns_d = getattr(self.adata, self.CN_ENRICHMENT_ATTR)

        enrichment_results = uns_d.get(self.CN_ENRICHMENT_KEY, None)

        # Choose a k
        chosen_k = self.choose_K.value

        if self.choose_K.value is not None:
            data = enrichment_results[str(chosen_k)].T.sort_index()
            cmap="bwr"
            mag_max = data.abs().max().max()
            self.enrichment_matrix_canvas.plot(
                data=data,
                cmap=cmap,
                vmin=-mag_max,
                vmax=mag_max,
                vcenter=0,
                figsize=(6,5) # default?
            )
        else:
            self.enrichment_matrix_canvas.clear()
    
        # Get values
class NolanWidget(QTabWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata):
        super().__init__()
        self.viewer = viewer
        
        self.compute_tab = NolanComputeWidget(self.viewer, adata)
        self.addTab(self.compute_tab.native, "Compute")

        self.plot_tab = NolanPlotWidget(self.viewer, adata)
        self.addTab(self.plot_tab, "Visualise")

class SpatialAnalysisTab(QTabWidget):
    """ Spatial Analysis classes; 1) Squidpy Wrapper, 2) General Wrapper """
    def __init__(self, viewer: "napari.viewer.Viewer", adata, subsetter):
        super().__init__()
        self.viewer = viewer
        self.subsetter = subsetter

        self.graph_builder = GraphBuilderWidget(self.viewer, adata)
        self.graph_builder.max_height = 400
        self.addTab(self.graph_builder.native, "Build Graph")

        self.nolan_cn = NolanWidget(self.viewer, adata)
        self.nolan_cn.max_height = 400
        self.addTab(self.nolan_cn, "Cellular Neighborhoods")

class FeatureModellingTab(QTableWidget):
    def __init__(self, viewer: "napari.viewer.Viewer", adata, subsetter):
        super().__init__()
        self.viewer = viewer
        self.subsetter = subsetter 

    # Widgets for visualising features;
    # Maybe extend spatialdata-view; 
    # matplotlib hist 
    # scanpy plots again

class AnalysisParentWidget(QWidget):
    """ UI tabs. """
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer
        self.meta_adata = None
        self.meta_sdata = None
        self.events = EmitterGroup(
            source=self, 
            meta_sdata_changed=None,
            meta_adata_changed=None
        )

        # If initial selection is valid, update 
        init_selected = viewer.layers.selection.active
        if init_selected is not None and "sdata" in init_selected.metadata:
            self.update_sdata_model()

        #self.viewer.layers.events.inserted.connect(self.refresh_choices_from_image)
        #self.viewer = viewer
        
        # Maintain adata state of current selected layer
        # self.loader = LoaderWidget(self.viewer)
        # self.loader.max_height = 200
        # self.addTab(self.loader.native, "Loader") 
        self.viewer.layers.selection.events.changed.connect(
            self.update_sdata_model)
        
        self.events.meta_sdata_changed.connect(self.refresh_adata_choices)
        self.events.meta_sdata_changed.connect(
            lambda x: AnnDataOperatorWidget.update_sdata_all_operators(x.value))
        
        self.layout = QVBoxLayout()
        
        # self._adata_selection = ComboBox(
        #     name="LayersWithContainedAdata",
        #     choices=self.get_adata_in_sdata,
        #     label="Select a contained adata",
        # )
        # self._adata_selection.scrollable = True
        # self._adata_selection.changed.connect(self.update_adata_model)
        # self.layout.addWidget(self._adata_selection.native)
        # Link adata_selection to the Tables annotating layer..
    

        # Parent Data Manager; Hold the memory reference to adatas in this class
        # On creation, empty
        self.subsetter = AnnDataSubsetterWidget(self.viewer, None)
        self.subsetter.min_height = 200
        self.subsetter.max_height = 500
        self.layout.addWidget(self.subsetter.native)
        # When the hotspot changes; update the tree
        self.events.meta_adata_changed.connect(
            lambda x: self.subsetter.create_model(x.value)) # Create new tree

        # When adata changes, update all operators
        self.subsetter.events.adata_created.connect(
            lambda x: AnnDataOperatorWidget.create_model_all_operators(x.value))

        self.subsetter.events.adata_changed.connect(
            lambda x: AnnDataOperatorWidget.update_model_all_operators(x.value))
        
        self.subsetter.events.adata_saved.connect(
            lambda x: self.save_adata_to_sdata(x.value)
        )
        
        # Hotdesk Adata
        adata = self.subsetter.adata

        self.tabs = QTabWidget()

        self.cell_typing_tab = CellTypingTab(
            viewer, adata, self.subsetter)
        
        self.spatial_analysis_tab = SpatialAnalysisTab(
            viewer, adata, self.subsetter)
        
        self.feature_modelling_tab = FeatureModellingTab(
            viewer, adata, self.subsetter)
        
        self.tabs.addTab(self.cell_typing_tab, "Cell Typing")
        self.tabs.addTab(self.spatial_analysis_tab, "Spatial Analysis")
        self.tabs.addTab(self.feature_modelling_tab, "Feature Modelling")

        self.layout.addWidget(self.tabs)
        self.setLayout(self.layout)

        # init
        if self._adata_selection.value is not None:
            self.update_adata_model()
        

    def get_adata_in_sdata(self, widget=None):
        if self.meta_sdata is not None:
            return list(self.meta_sdata.tables.keys())
        else:
            return []
        
    def get_layers_with_valid_contained_sdata(self, widget=None):
        # Reference to the sdata in ithe main mutliscale image 
        return [
            l.name for l in self.viewer.layers 
                if isinstance(
                    l.data, napari.layers._multiscale_data.MultiScaleData) and
                "sdata" in l.metadata and
                l.metadata["sdata"] is not None and
                l.metadata["sdata"].is_backed() and
                "adata" in l.metadata and 
                l.metadata["adata"] is not None and 
                l.metadata["adata"].shape[0] > 0#and isinstance(l, napari.layers.Labels)
            ]
    
    def get_layers_with_contained_adata(self, widget=None):
        layers = [
            l.name for l in self.viewer.layers 
                if "adata" in l.metadata and 
                l.metadata["adata"] is not None and 
                l.metadata["adata"].shape[0] > 0#and isinstance(l, napari.layers.Labels)
            ]
        
        if layers is None:
            raise AttributeError("No layers with contained adata found.")
        
        return layers

    def is_valid_selection(self, selected):
        return selected is not None \
            and isinstance(
                selected.data, 
                napari.layers._multiscale_data.MultiScaleData) \
            and "sdata" in selected.metadata \
            and selected.metadata["sdata"] is not None \
            and selected.metadata["sdata"].is_backed()

    def update_sdata_model(self):
        selected = self.viewer.layers.selection.active
        sdata = None
        if self.is_valid_selection(selected):
            sdata = selected.metadata["sdata"]

        # If we have a new sdata, update
        if sdata is not None:
            if self.meta_sdata is None or self.meta_sdata is not sdata:
                self.meta_sdata = sdata
                self.events.meta_sdata_changed(value=self.meta_sdata)
                
        # layer = get_selected_layer(self.viewer, self._adata_parent_selection)
        # self.layer = layer
        # self.meta_adata = layer.metadata["adata"]
        # self.layer.metadata["adata_processor_widget"] = self
        # self.events.meta_adata_changed(value=self.meta_adata)
    
    def update_adata_model(self):
        selection = self._adata_selection.value
        self.meta_adata = self.meta_sdata[selection]
        self.events.meta_adata_changed(value=self.meta_adata)

    def refresh_adata_choices(self):
        self._adata_selection.reset_choices()

    def save_adata_to_sdata(self, new_adata):
        selection = self._adata_selection.value
        #meta_adata = self.meta_sdata[selection]
        # Dont overwrite, track original?
        new_selection_label = make_unique_sdata_element_name(
            selection + "_post")
        # Can parse AnnData directly as it will inherit the attrs from uns
        self.meta_sdata[new_selection_label] = new_adata
        self.meta_sdata.write_element(new_selection_label)
        self._adata_selection.reset_choices()
        
    # def _add_dearrayer(self):
    #     self.dearrayer = TMADearrayerNapariWidget(self.viewer)
    #     self.dearrayer.max_height = 200
    #     self.addTab(self.dearrayer.native, "> Dearrayer")
    #     self.dearrayer._dearray_button.changed.connect(self._add_segmenter)

    # def _add_segmenter(self):
    #     self.segmenter = TMASegmenterNapariWidget(self.viewer)
    #     self.segmenter.max_height = 700
    #     self.segmenter.max_width = 500
    #     self.addTab(self.segmenter.native, "> Segmenter")
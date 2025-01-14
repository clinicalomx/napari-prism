import napari
from anndata import AnnData
from magicgui.widgets import ComboBox

from napari_prism.widgets.adata_ops._base_widgets import AnnDataOperatorWidget


class ObsAggregatorWidget(AnnDataOperatorWidget):
    """Interface for using the ObsAggregator class."""

    def __init__(self, viewer: "napari.viewer.Viewer", adata: AnnData) -> None:
        super().__init__(viewer, adata)

    def create_parameter_widgets(self) -> None:
        """Create widgets for exposing the functions of the ObsAggregator class."""
        self.sample_key = ComboBox(
            name="SampleKey",
            choices=self.get_categorical_obs_keys,
            label="Sample-Level Key",
        )

"""Mainly wrappers for spatialdata_plot; simplified user plotting."""

from typing import Literal

import dask.array as da
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns
import skimage
import xarray as xr
from anndata import AnnData
from spatialdata import SpatialData

from napari_prism.models.tma_ops._tma_image import TMASegmenter


def image(
    sdata: SpatialData,
    image_name: str,
    channel_label: str = "DAPI",
    channel_cmap: str = "gray",
    alpha: float = 1.0,
    figsize: tuple[int, int] | None = None,
    dpi: int | None = None,
    coordinate_system: str = "global",  # If generated with prism.
) -> None | SpatialData:
    sdata.pl.render_images(
        image_name, channel_label, cmap=channel_cmap, alpha=alpha
    ).pl.show(figsize=figsize, dpi=dpi, coordinate_systems=coordinate_system)


def mask_tma(
    sdata: SpatialData,
    label_name: str,
    image_name: str | None = None,
    image_channel_label: str | None = None,
    image_channel_cmap: str | None = None,
    image_alpha: float | None = None,
    figsize: tuple[int, int] | None = None,
    dpi: int | None = None,
    coordinate_system: str = "global",  # If generated with prism.
    **kwargs,
):
    """
    Plot masks.

    Args:
        sdata: SpatialData object.
        label_name: Name of the label to plot.
        image_name (Optional): Name of the image to plot.
        image_channel_label (Optional): Label of the channel to plot. Must be
            provided if `image_name` is provided.
        image_channel_cmap (Optional): Colormap of the channel to plot.
        image_alpha (Optional): Alpha of the channel to plot.
        **kwargs: Passed to `PlotAccessor.render_labels`.
    """
    if image_name is not None and image_channel_label is not None:
        sdata.pl.render_images(
            image_name,
            image_channel_label,
            cmap=image_channel_cmap,
            alpha=image_alpha,
        ).pl.render_labels(label_name, **kwargs).pl.show(
            figsize=figsize, dpi=dpi, coordinate_systems=coordinate_system
        )
    else:
        sdata.pl.render_labels(label_name, **kwargs).pl.show(
            figsize=figsize, dpi=dpi, coordinate_systems=coordinate_system
        )


def dearray_tma():
    pass


def _apply_skimage_to_dataarray(function, dataarray):
    dataarray.data = function(da.array(dataarray.data))
    return dataarray


def dataarray_to_rgb(
    dataarray: xr.DataArray,
    color_order: list[str],
    auto_contrast: bool = True,
):
    if "c" not in dataarray.dims:
        raise ValueError("Input DataArray must have a 'c' dimension.")

    c_size = dataarray.sizes["c"]
    rgb_order_map = {
        "R": 0,
        "G": 1,
        "B": 2,
    }
    if c_size == 1:
        # Create an RGB image dependign on the c_color_order
        signal_channel = dataarray.isel(c=0)
        blank_channel = None
        # populate depending on color order;
        signal_color = color_order[0]
        im_order = [blank_channel, blank_channel, blank_channel]
        im_order[rgb_order_map[signal_color]] = signal_channel

    elif c_size == 2:
        signal_channel = dataarray.isel(c=0)
        second_signal_channel = dataarray.isel(c=1)
        blank_channel = None
        signal_color = color_order[0]
        second_signal_color = color_order[1]
        im_order = [blank_channel, blank_channel, blank_channel]
        im_order[rgb_order_map[signal_color]] = signal_channel
        im_order[rgb_order_map[second_signal_color]] = second_signal_channel

    elif c_size == 3:
        signal_channel = dataarray.isel(c=0)
        second_signal_channel = dataarray.isel(c=1)
        third_signal_channel = dataarray.isel(c=2)
        blank_channel = None
        signal_color = color_order[0]
        second_signal_color = color_order[1]
        third_signal_color = color_order[2]
        im_order = [blank_channel, blank_channel, blank_channel]
        im_order[rgb_order_map[signal_color]] = signal_channel
        im_order[rgb_order_map[second_signal_color]] = second_signal_channel
        im_order[rgb_order_map[third_signal_color]] = third_signal_channel

    else:
        raise ValueError(
            "DataArray must have 1, 2, or 3 channels in the 'c' dimension."
        )
    if auto_contrast:
        im_order = [
            (
                _apply_skimage_to_dataarray(
                    skimage.exposure.equalize_adapthist, x
                )
                if x is not None
                else x
            )
            for x in im_order
        ]
    blank_channel_arr = xr.zeros_like(dataarray.isel(c=0))
    im_order = [blank_channel_arr if x is None else x for x in im_order]
    rgb_image = xr.concat(im_order, dim="c")
    rgb_image = rgb_image.assign_coords(c=["R", "G", "B"])
    rgb_image = rgb_image.transpose("y", "x", "c")
    return rgb_image


def preview_tma_segmentation(
    sdata: SpatialData,
    image_name: str,
    segmentation_channel: str | list[str],
    color_order: list[str],
    channel_merge_method: Literal["max", "mean", "min", "sum"] = "max",
    optional_nuclear_channel: str | None = None,
    reference_coordinate_system: str = "global",
    auto_contrast: bool = True,
):
    model = TMASegmenter(
        sdata=sdata,
        image_name=image_name,
        reference_coordinate_system=reference_coordinate_system,
    )

    arr_preview = model.segment_all(
        scale="scale0",
        tiling_shapes=None,
        model_type="nuclei",
        nuclei_diam_um=None,
        segmentation_channel=segmentation_channel,
        channel_merge_method=channel_merge_method,
        optional_nuclear_channel=optional_nuclear_channel,
        preview=True,
    )

    rgb_image = dataarray_to_rgb(arr_preview, color_order, auto_contrast)
    plt.imshow(rgb_image)


def umap(*args, **kwargs):
    sc.pl.umap(*args, **kwargs)


def tsne(*args, **kwargs):
    sc.pl.tsne(*args, **kwargs)


def pca(*args, **kwargs):
    sc.pl.pca(*args, **kwargs)


def cluster_scores(
    input_data: AnnData | pd.DataFrame,
    clustering_score: Literal["ARI", "NMI", "AMI"] = "ARI",
    **kwargs,
):
    if isinstance(input_data, AnnData):
        # Look for the cluster scores in AnnData.uns keys;
        matches = [
            x
            for x in input_data.uns_keys()
            if f"{clustering_score}_cluster_scores" in x
        ]

        if len(matches) != 1:
            raise ValueError(
                "Could not find any unique cluster scores in AnnData.uns."
            )
        else:
            cluster_scores = input_data.uns[matches[0]]
    else:
        cluster_scores = input_data

    # Plot the cluster scores;
    sns.heatmap(cluster_scores, **kwargs)

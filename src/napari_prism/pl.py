"""Mainly wrappers for spatialdata_plot; simplified user plotting."""

from typing import Literal

import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import skimage
import xarray as xr
from anndata import AnnData
from spatialdata import SpatialData
from spatialdata.transformations import (
    get_transformation_between_coordinate_systems,
)

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
    show: bool = True,
) -> None | SpatialData:
    """
    Displays an Image element from the SpatialData object.

    Args:
        sdata: The SpatialData object containing the image.
        image_name: The name of the image to display.
        channel_label: The label of the channel to display.
        channel_cmap: The colormap to use for the channel.
        alpha: The alpha value for the image.
        figsize: The size of the figure.
        dpi: The DPI of the figure.
        coordinate_system: The coordinate system to use for rendering.
        show: Whether to show the plot or return it.

    Returns:
        None if show is True, otherwise returns the rendered SpatialData
        object.
    """
    out = sdata.pl.render_images(
        image_name, channel_label, cmap=channel_cmap, alpha=alpha
    )
    if show:
        out.pl.show(
            figsize=figsize, dpi=dpi, coordinate_systems=coordinate_system
        )
    else:
        return out


def mask_tma(
    sdata: SpatialData,
    label_name: str,
    image_name: str | None = None,
    channel_label: str | None = None,
    channel_cmap: str | None = "gray",
    figsize: tuple[int, int] | None = None,
    dpi: int | None = None,
    coordinate_system: str = "global",  # If generated with prism.
    show: bool = True,
    **kwargs,
) -> None | SpatialData:
    """
    Displays a labels element from the SpatialData object, such as the generated
    TMA masks.

    Args:
        sdata: The SpatialData object containing the image.
        label_name: The name of the label to display.
        image_name: The name of the image to display.
        channel_label: The label of the channel to display.
        channel_cmap: The colormap to use for the channel.
        figsize: The size of the figure.
        dpi: The DPI of the figure.
        coordinate_system: The coordinate system to use for rendering.
        show: Whether to show the plot or return it.
        **kwargs: Passed to spatialdata.pl.render_labels.

    Returns:
        None if show is True, otherwise returns the rendered SpatialData object.
    """
    IMALPHA = 1.0  # we have to define it here despite being the default above?
    if image_name is not None and channel_label is not None:
        out = image(
            sdata, image_name, channel_label, channel_cmap, IMALPHA, show=False
        )
    else:
        out = sdata

    out = out.pl.render_labels(label_name, **kwargs)

    if show:
        out.pl.show(
            coordinate_systems=coordinate_system,
            figsize=figsize,
            dpi=dpi,
        )
    else:
        return out


def dearray_tma(
    sdata: SpatialData,
    shapes_name: str,
    image_name: str | None = None,
    channel_label: str | None = None,
    channel_cmap: str | None = "gray",
    figsize: tuple[int, int] | None = None,
    dpi: int | None = None,
    coordinate_system: str = "global",  # If generated with prism.
    fill_alpha: float = 0.0,
    outline_alpha: float = 1.0,
    outline_width: float = 2.0,
    outline_color: str = "blue",
    tma_annotation_color: str = "white",
    tma_annotation_fontsize: float = 20,
    show: bool = True,
    **kwargs,
) -> None | SpatialData:
    """
    Displays a shapes element from the SpatialData object, such as the
    generated TMA envelopes and cores.

    Args:
        sdata: The SpatialData object containing the image.
        shapes_name: The name of the shapes to display.
        image_name: The name of the image to display.
        channel_label: The label of the channel to display.
        channel_cmap: The colormap to use for the channel.
        figsize: The size of the figure.
        dpi: The DPI of the figure.
        coordinate_system: The coordinate system to use for rendering.
        fill_alpha: Alpha value for filling shapes.
        outline_alpha: Alpha value for shape outlines.
        outline_width: Width of shape outlines.
        outline_color: Color of shape outlines.
        tma_annotation_color: Color of TMA annotations.
        tma_annotation_fontsize: Font size of TMA annotations.
        show: Whether to show the plot or return it.
        **kwargs: Passed to spatialdata.pl.render_shapes.

    Returns:
        None if show is True, otherwise returns the rendered SpatialData
        object.
    """
    IMALPHA = 1.0
    if image_name is not None and channel_label is not None:
        out = image(
            sdata, image_name, channel_label, channel_cmap, IMALPHA, show=False
        )
    else:
        out = sdata

    out = out.pl.render_shapes(
        shapes_name,
        fill_alpha=fill_alpha,
        outline_alpha=outline_alpha,
        outline_width=outline_width,
        outline_color=outline_color,
        **kwargs,
    )

    def _transform_point(point, affine_matrix):
        homogeneous_point = np.array([point[0], point[1], 1])

        transformed = affine_matrix @ homogeneous_point

        if transformed[2] != 0:
            transformed /= transformed[2]

        return transformed[:2]

    if show:
        out.pl.show(
            coordinate_systems=coordinate_system,
            figsize=figsize,
            dpi=dpi,
        )
        # Annotate text;
        transforms = get_transformation_between_coordinate_systems(
            sdata, sdata[shapes_name], coordinate_system
        )
        affine = transforms.to_affine_matrix(("x", "y"), ("x", "y"))
        for _, geom in sdata[shapes_name].iterrows():
            xmin, ymin, xmax, ymax = geom["geometry"].bounds
            middle = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            middle_global = tuple(_transform_point(middle, affine))
            plt.annotate(
                geom["tma_label"],
                xy=middle_global,
                color=tma_annotation_color,
                fontsize=tma_annotation_fontsize,
                ha="center",
                va="center",
            )
    else:
        return out


def segment_tma(
    sdata: SpatialData,
    segmentation_name: str,
    image_name: str | None = None,
    channel_label: str | None = None,
    channel_cmap: str | None = "gray",
    fill_alpha: float = 0.0,
    outline_alpha: float = 1.0,
    contour_px: int = 3,
    figsize: tuple[int, int] | None = None,
    dpi: int | None = None,
    coordinate_system: str = "global",  # If generated with prism.
    show: bool = True,
    **kwargs,
):
    """
    Displays a labels element from the SpatialData object, such as the
    cell segmentation results.

    Args:
        sdata: The SpatialData object containing the image.
        segmentation_name: The name of the segmentation to display.
        image_name: The name of the image to display.
        channel_label: The label of the channel to display.
        channel_cmap: The colormap to use for the channel.
        fill_alpha: Alpha value for filling shapes.
        outline_alpha: Alpha value for shape outlines.
        contour_px: Pixel width of contours.
        figsize: The size of the figure.
        dpi: The DPI of the figure.
        coordinate_system: The coordinate system to use for rendering.
        show: Whether to show the plot or return it.
        **kwargs: Passed to spatialdata.pl.render_labels.

    Returns:
        None if show is True, otherwise returns the rendered SpatialData
        object.
    """
    IMALPHA = 1.0
    if image_name is not None and channel_label is not None:
        out = image(
            sdata, image_name, channel_label, channel_cmap, IMALPHA, show=False
        )
    else:
        out = sdata

    out = out.pl.render_labels(
        segmentation_name,
        fill_alpha=fill_alpha,
        outline_alpha=outline_alpha,
        contour_px=contour_px,
        **kwargs,
    )

    if show:
        out.pl.show(
            coordinate_systems=coordinate_system,
            figsize=figsize,
            dpi=dpi,
        )

    else:
        return out


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
) -> None:
    """
    Displays a visual preview of the input data given to Cellpose for
    segmentation. This is useful for checking if the provided markers provide a
    good representation for nuclei, cell membranes, cytoplasms, etc.

    Args:
        sdata: The SpatialData object containing the image.
        image_name: The name of the image to display.
        segmentation_channel: The channel(s) to use for segmentation.
        color_order: The order of colors for the RGB image.
        channel_merge_method: Method to merge channels. Options are "max",
            "mean", "min", or "sum".
        optional_nuclear_channel: Optional nuclear channel for segmentation.
        reference_coordinate_system: Coordinate system to use for rendering.
        auto_contrast: Whether to apply auto contrast to the image.
    """
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
) -> None:
    """
    Plot the cluster stability scores from a clustering run.

    Args:
        input_data: AnnData object or DataFrame containing clustering scores.
        clustering_score: The clustering score to plot. Options are "ARI",
            "NMI", or "AMI".
        **kwargs: Additional keyword arguments for the heatmap.
    """
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
